import { randomUUID } from "node:crypto"
import { chmod, lstat, mkdir, open, readFile, rename, rm, unlink } from "node:fs/promises"
import { basename, dirname, join } from "node:path"
import type { OAuthCredentials, OAuthProviderInterface } from "../oauth/types.js"

export type StoredCredential = {
  readonly provider: string
  readonly credentials: OAuthCredentials
  readonly updatedAtMs: number
}

export interface CredentialStore {
  get(provider: string): Promise<StoredCredential | undefined>
  set(provider: string, credentials: OAuthCredentials): Promise<StoredCredential>
  delete(provider: string): Promise<void>
}

export class CredentialStoreCorruptError extends Error {
  readonly path: string

  constructor(path: string, cause: unknown) {
    super(`credential_store_corrupt:${path}`, { cause })
    this.name = "CredentialStoreCorruptError"
    this.path = path
  }
}

export class InMemoryCredentialStore implements CredentialStore {
  readonly #items = new Map<string, StoredCredential>()

  async get(provider: string): Promise<StoredCredential | undefined> {
    return this.#items.get(provider)
  }

  async set(provider: string, credentials: OAuthCredentials): Promise<StoredCredential> {
    const stored = { provider, credentials, updatedAtMs: Date.now() }
    this.#items.set(provider, stored)
    return stored
  }

  async delete(provider: string): Promise<void> {
    this.#items.delete(provider)
  }
}

export class FileCredentialStore implements CredentialStore {
  readonly #path: string
  readonly #legacyPaths: readonly string[]

  constructor(path: string, legacyPaths: readonly string[] = []) {
    this.#path = path
    this.#legacyPaths = legacyPaths
  }

  async get(provider: string): Promise<StoredCredential | undefined> {
    const items = await this.#readAll()
    return items[provider]
  }

  async set(provider: string, credentials: OAuthCredentials): Promise<StoredCredential> {
    const items = await this.#readAll()
    const stored = { provider, credentials, updatedAtMs: Date.now() }
    items[provider] = stored
    await this.#writeAll(items)
    return stored
  }

  async delete(provider: string): Promise<void> {
    const items = await this.#readAll()
    delete items[provider]
    if (Object.keys(items).length === 0) {
      await rm(this.#path, { force: true })
      return
    }
    await this.#writeAll(items)
  }

  async #readAll(): Promise<Record<string, StoredCredential>> {
    const primary = await this.#readExisting(this.#path)
    if (primary !== undefined) {
      return primary
    }
    for (const legacyPath of this.#legacyPaths) {
      const legacy = await this.#readExisting(legacyPath)
      if (legacy !== undefined && Object.keys(legacy).length > 0) {
        return legacy
      }
    }
    return {}
  }

  async #readExisting(path: string): Promise<Record<string, StoredCredential> | undefined> {
    try {
      await refuseSymlink(path)
      return parseCredentialRecord(await readFile(path, "utf8"))
    } catch (error) {
      if (isNotFound(error)) {
        return undefined
      }
      if (error instanceof SyntaxError) {
        throw new CredentialStoreCorruptError(path, error)
      }
      throw error
    }
  }

  async #writeAll(items: Record<string, StoredCredential>): Promise<void> {
    const directory = dirname(this.#path)
    await mkdir(directory, { recursive: true })
    await refuseSymlink(this.#path)
    const tempPath = join(directory, `.${basename(this.#path)}.${process.pid}.${randomUUID()}.tmp`)
    const handle = await open(tempPath, "wx", 0o600)
    let shouldRemoveTemp = true
    try {
      await handle.writeFile(`${JSON.stringify(items, null, 2)}\n`, "utf8")
      await handle.sync()
      await handle.close()
      await chmod(tempPath, 0o600)
      await refuseSymlink(this.#path)
      if (process.env["ATOMISTIC_TEST_CREDENTIAL_STORE_FAIL_BEFORE_REPLACE"] === "1") {
        throw new Error("credential_store_replace_interrupted")
      }
      await rename(tempPath, this.#path)
      await fsyncDirectory(directory)
      shouldRemoveTemp = false
    } finally {
      await handle.close().catch(error => {
        if (!isBadFileDescriptor(error)) {
          throw error
        }
      })
      if (shouldRemoveTemp) {
        await unlink(tempPath).catch(error => {
          if (!isNotFound(error)) {
            throw error
          }
        })
      }
    }
  }
}

export class CredentialManager {
  readonly #store: CredentialStore

  constructor(store: CredentialStore) {
    this.#store = store
  }

  async saveOAuthCredentials(provider: string, credentials: OAuthCredentials): Promise<StoredCredential> {
    return await this.#store.set(provider, credentials)
  }

  async resolveApiKey(provider: OAuthProviderInterface, nowMs = Date.now()): Promise<string> {
    const stored = await this.#store.get(provider.id)
    if (stored === undefined) {
      throw new Error(`missing_credentials:${provider.id}`)
    }
    const credentials = await this.#refreshIfNeeded(provider, stored.credentials, nowMs)
    if (credentials !== stored.credentials) {
      await this.#store.set(provider.id, credentials)
    }
    return provider.getApiKey?.(credentials) ?? credentials.access
  }

  async #refreshIfNeeded(
    provider: OAuthProviderInterface,
    credentials: OAuthCredentials,
    nowMs: number,
  ): Promise<OAuthCredentials> {
    if (credentials.expires > nowMs + 30_000) {
      return credentials
    }
    if (provider.refreshToken === undefined) {
      throw new Error(`credential_expired_without_refresh:${provider.id}`)
    }
    return await provider.refreshToken(credentials)
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT"
}

function isBadFileDescriptor(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "EBADF"
}

async function refuseSymlink(path: string): Promise<void> {
  try {
    const stat = await lstat(path)
    if (stat.isSymbolicLink()) {
      throw new Error("credential_store_symlink_refused")
    }
  } catch (error) {
    if (isNotFound(error)) {
      return
    }
    throw error
  }
}

async function fsyncDirectory(path: string): Promise<void> {
  const handle = await open(path, "r")
  try {
    await handle.sync()
  } finally {
    await handle.close()
  }
}

function parseCredentialRecord(raw: string): Record<string, StoredCredential> {
  const value: unknown = JSON.parse(raw)
  if (!isRecord(value)) {
    return {}
  }
  const items: Record<string, StoredCredential> = {}
  for (const [provider, candidate] of Object.entries(value)) {
    if (isStoredCredential(candidate)) {
      items[provider] = candidate
    }
  }
  return items
}

function isStoredCredential(value: unknown): value is StoredCredential {
  if (!isRecord(value)) {
    return false
  }
  const credentials = value["credentials"]
  return (
    typeof value["provider"] === "string"
    && typeof value["updatedAtMs"] === "number"
    && isRecord(credentials)
    && typeof credentials["access"] === "string"
    && typeof credentials["expires"] === "number"
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}
