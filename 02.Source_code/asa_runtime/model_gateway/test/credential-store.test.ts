import assert from "node:assert/strict"
import { lstat, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"
import { FileCredentialStore } from "../src/index.js"
import type { OAuthCredentials } from "../src/index.js"

test("credential-store-atomic writes provider credentials by atomic replace", async () => {
  // Given: a credential file store in a temporary directory.
  const root = await mkdtemp(join(tmpdir(), "atomistic-credential-store-atomic-"))
  const credentialPath = join(root, "credentials.json")
  const store = new FileCredentialStore(credentialPath)

  try {
    // When: provider credentials are stored.
    await store.set("openai-codex", credentials("atomic-access"))

    // Then: the final file is private and contains valid credentials.
    const mode = (await lstat(credentialPath)).mode & 0o777
    const payload = jsonRecord(await readFile(credentialPath, "utf8"))
    assert.equal(mode, 0o600)
    assert.equal(storedAccess(payload, "openai-codex"), "atomic-access")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test("credential-store-atomic preserves previous store when replace is interrupted", async () => {
  // Given: an existing store and a configured replace interruption.
  const root = await mkdtemp(join(tmpdir(), "atomistic-credential-store-partial-"))
  const credentialPath = join(root, "credentials.json")
  const previousFailureHook = process.env["ATOMISTIC_TEST_CREDENTIAL_STORE_FAIL_BEFORE_REPLACE"]
  process.env["ATOMISTIC_TEST_CREDENTIAL_STORE_FAIL_BEFORE_REPLACE"] = "1"
  const store = new FileCredentialStore(credentialPath)
  await writeFile(
    credentialPath,
    `${JSON.stringify({ "openai-codex": stored("openai-codex", "old-access") })}\n`,
    { encoding: "utf8", mode: 0o600 },
  )

  try {
    // When: a new write is interrupted before replacement.
    await assert.rejects(
      store.set("openai-codex", credentials("new-access")),
      error => error instanceof Error && error.message === "credential_store_replace_interrupted",
    )

    // Then: previous credentials remain readable and no temp secret remains.
    const payload = jsonRecord(await readFile(credentialPath, "utf8"))
    assert.equal(storedAccess(payload, "openai-codex"), "old-access")
    assert.deepEqual(await directoryNames(root), ["credentials.json"])
  } finally {
    restoreFailureHook(previousFailureHook)
    await rm(root, { recursive: true, force: true })
  }
})

test("credential-store-atomic rejects corrupt credential stores", async () => {
  // Given: a corrupt existing store.
  const root = await mkdtemp(join(tmpdir(), "atomistic-credential-store-corrupt-"))
  const credentialPath = join(root, "credentials.json")
  const store = new FileCredentialStore(credentialPath)
  await writeFile(credentialPath, "{not-json}\n", "utf8")

  try {
    // When / Then: reading credentials raises a typed corrupt-store error.
    await assert.rejects(
      store.get("openai-codex"),
      error => error instanceof Error && error.name === "CredentialStoreCorruptError",
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test("credential-store-atomic refuses symlink credential path", async t => {
  // Given: the credential path is a symlink.
  const root = await mkdtemp(join(tmpdir(), "atomistic-credential-store-symlink-"))
  const target = join(root, "target.json")
  const credentialPath = join(root, "credentials.json")
  await writeFile(target, "{}\n", "utf8")

  try {
    try {
      await symlink(target, credentialPath)
    } catch (error) {
      if (isNodeError(error) && error.code === "EPERM") {
        t.skip("symlink unsupported")
        return
      }
      throw error
    }
    const store = new FileCredentialStore(credentialPath)

    // When / Then: storing credentials refuses the symlink and leaves the target unchanged.
    await assert.rejects(
      store.set("openai-codex", credentials("symlink-access")),
      error => error instanceof Error && error.message === "credential_store_symlink_refused",
    )
    assert.equal(await readFile(target, "utf8"), "{}\n")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

function credentials(access: string): OAuthCredentials {
  return { access, refresh: `${access}-refresh`, expires: 4_102_444_800_000 }
}

function stored(provider: string, access: string): { readonly provider: string; readonly credentials: OAuthCredentials; readonly updatedAtMs: number } {
  return { provider, credentials: credentials(access), updatedAtMs: 1 }
}

function jsonRecord(raw: string): Record<string, unknown> {
  const value: unknown = JSON.parse(raw)
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("json_object_required")
  }
  return Object.fromEntries(Object.entries(value))
}

function storedAccess(payload: Record<string, unknown>, provider: string): string {
  const value = payload[provider]
  assertStoredCredentialRecord(value)
  return value.credentials.access
}

async function directoryNames(root: string): Promise<readonly string[]> {
  return (await readdir(root)).sort()
}

function assertStoredCredentialRecord(
  value: unknown,
): asserts value is { readonly credentials: { readonly access: string } } {
  assert.equal(typeof value, "object")
  assert.notEqual(value, null)
  assert.equal(Array.isArray(value), false)
  assert.equal(typeof recordField(recordField(value, "credentials"), "access"), "string")
}

function recordField(value: unknown, field: string): unknown {
  if (!isRecord(value)) {
    throw new TypeError("record_required")
  }
  return value[field]
}

function restoreFailureHook(value: string | undefined): void {
  if (value === undefined) {
    delete process.env["ATOMISTIC_TEST_CREDENTIAL_STORE_FAIL_BEFORE_REPLACE"]
    return
  }
  process.env["ATOMISTIC_TEST_CREDENTIAL_STORE_FAIL_BEFORE_REPLACE"] = value
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}
