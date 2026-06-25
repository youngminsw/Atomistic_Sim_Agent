import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { AddressInfo } from "node:net"
import test from "node:test"
import {
  buildGatewayAdapterResponse,
  CredentialManager,
  FakeOAuthProvider,
  FileCredentialStore,
  InMemoryCredentialStore,
  createModelGatewayServer,
} from "../src/index.js"
import type { OAuthAuthInfo } from "../src/index.js"

test("CredentialManager refreshes expired OAuth credentials through provider boundary", async () => {
  const provider = new FakeOAuthProvider({
    id: "openai-codex",
    accessToken: "access:initial",
    refreshToken: "refresh:initial",
    expiresInMs: -1,
  })
  const store = new InMemoryCredentialStore()
  const manager = new CredentialManager(store)
  const credentials = await provider.login({
    onAuth: () => undefined,
    onPrompt: async () => "",
  })
  await manager.saveOAuthCredentials(provider.id, credentials)

  const apiKey = await manager.resolveApiKey(provider)
  const stored = await store.get(provider.id)

  assert.equal(apiKey, "access:initial:refreshed")
  assert.equal(stored?.credentials.access, "access:initial:refreshed")
})

test("FileCredentialStore persists credentials for gateway restarts", async () => {
  const root = await mkdtemp(join(tmpdir(), "atomistic-gateway-store-"))
  const path = join(root, "credentials.json")
  try {
    const first = new FileCredentialStore(path)
    await first.set("openai-codex", {
      access: "access:persisted",
      refresh: "refresh:persisted",
      expires: 123,
    })

    const second = new FileCredentialStore(path)
    const stored = await second.get("openai-codex")

    assert.equal(stored?.provider, "openai-codex")
    assert.equal(stored?.credentials.access, "access:persisted")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test("FakeOAuthProvider exposes login auth info without network", async () => {
  let authInfo: OAuthAuthInfo | undefined
  const provider = new FakeOAuthProvider({
    id: "openai-codex",
    name: "ChatGPT OAuth Test",
    authUrl: "https://auth.openai.com/oauth/authorize",
    accountId: "acct-1",
    email: "agent@example.test",
  })

  const credentials = await provider.login({
    onAuth: info => {
      authInfo = info
    },
    onPrompt: async () => "",
  })

  assert.equal(authInfo?.url, "https://auth.openai.com/oauth/authorize")
  assert.equal(credentials.accountId, "acct-1")
  assert.equal(credentials.email, "agent@example.test")
})

test("buildGatewayAdapterResponse returns Python-compatible snake_case adapter", () => {
  const response = buildGatewayAdapterResponse({
    model_provider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "oauth",
      auth_refresh_command: "model-gateway auth refresh --print",
    },
    credential: {
      access: "gateway-access",
      refresh: "gateway-refresh",
      expires: 42,
    },
  })

  assert.equal(response.ok, true)
  assert.equal(response.adapter.provider, "local_gateway")
  assert.equal(response.adapter.base_url, "http://127.0.0.1:8787/v1")
  assert.equal(response.adapter.auth_refresh_command, "model-gateway auth refresh --print")
  assert.equal(response.api_key, "gateway-access")
  assert.equal(response.credential_expires, 42)
})

test("createModelGatewayServer exposes health, models, request id, and upstream forwarding", async () => {
  const upstream = createModelGatewayServer({
    modelProvider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:1/v1",
      auth_mode: "none",
    },
    requestIdFactory: () => "upstream-1",
  })
  await listen(upstream)
  const upstreamUrl = rootUrl(upstream)

  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "gateway",
      streaming: true,
    },
    upstreamBaseUrl: `${upstreamUrl}/v1`,
    requestIdFactory: () => "gw-test-1",
  })
  await listen(gateway)
  const gatewayUrl = rootUrl(gateway)

  try {
    const health = await getJson(`${gatewayUrl}/healthz`)
    const models = await getJson(`${gatewayUrl}/v1/models`)
    const response = await postJson(`${gatewayUrl}/v1/responses`, {
      model: "gpt-5.5",
      input: "route one task to QA",
    })

    assert.equal(health["ok"], true)
    assert.equal(health["auth_mode"], "gateway")
    assert.equal(models["gateway_request_id"], "gw-test-1")
    assert.equal(response["gateway_request_id"], "gw-test-1")
    assert.equal(response["gateway_provider"], "local_gateway")
    assert.equal(response["output_text"], "gateway_echo_ready")
  } finally {
    await close(gateway)
    await close(upstream)
  }
})

test("createModelGatewayServer reports missing credentials as hard gateway blocker", async () => {
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "gateway",
    },
    requestIdFactory: () => "gw-missing-credential",
  })
  await listen(gateway)
  try {
    const response = await fetch(`${rootUrl(gateway)}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-5.5", input: "hello" }),
    })
    const body = (await response.json()) as { error: { code: string }; gateway_request_id: string }

    assert.equal(response.status, 401)
    assert.equal(body.error.code, "missing_gateway_credentials")
    assert.equal(body.gateway_request_id, "gw-missing-credential")
  } finally {
    await close(gateway)
  }
})

async function listen(server: ReturnType<typeof createModelGatewayServer>): Promise<void> {
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
}

async function close(server: ReturnType<typeof createModelGatewayServer>): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.close(error => {
      if (error) {
        reject(error)
        return
      }
      resolve()
    })
  })
}

function rootUrl(server: ReturnType<typeof createModelGatewayServer>): string {
  const address = server.address() as AddressInfo
  return `http://127.0.0.1:${address.port}`
}

async function getJson(url: string): Promise<Record<string, unknown>> {
  const response = await fetch(url)
  assert.equal(response.status, 200)
  return (await response.json()) as Record<string, unknown>
}

async function postJson(url: string, payload: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    method: "POST",
    headers: { authorization: "Bearer gateway-token", "content-type": "application/json" },
    body: JSON.stringify(payload),
  })
  assert.equal(response.status, 200)
  return (await response.json()) as Record<string, unknown>
}
