import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname } from "node:path"
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

  constructor(path: string) {
    this.#path = path
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
    try {
      return JSON.parse(await readFile(this.#path, "utf8")) as Record<string, StoredCredential>
    } catch (error) {
      if (isNotFound(error)) {
        return {}
      }
      throw error
    }
  }

  async #writeAll(items: Record<string, StoredCredential>): Promise<void> {
    await mkdir(dirname(this.#path), { recursive: true })
    await writeFile(this.#path, `${JSON.stringify(items, null, 2)}\n`, { encoding: "utf8", mode: 0o600 })
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
