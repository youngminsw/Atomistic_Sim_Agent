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
  ModelProviderConfigError,
  createModelGatewayServer,
  parseModelProviderConfig,
  toPythonModelProviderAdapter,
} from "../src/index.js"
import type { OAuthAuthInfo } from "../src/index.js"

test("parseModelProviderConfig accepts explicit direct OpenAI provider config", () => {
  const config = parseModelProviderConfig({
    provider: "openai",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "https://api.openai.com/v1/",
    auth_mode: "api_key",
  })

  assert.equal(config.provider, "openai")
  assert.equal(config.baseUrl, "https://api.openai.com/v1")
  assert.equal(config.apiKeyEnv, "OPENAI_API_KEY")
  assert.deepEqual(toPythonModelProviderAdapter(config), {
    provider: "openai",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "https://api.openai.com/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: false,
    api_key_env: "OPENAI_API_KEY",
    auth_mode: "api_key",
  })
})

test("parseModelProviderConfig rejects low reasoning for physics decisions", () => {
  assert.throws(
    () =>
      parseModelProviderConfig({
        provider: "oauth_gateway",
        model: "gpt-5.3-codex-spark",
        reasoning_effort: "low",
        base_url: "http://127.0.0.1:8787/v1",
        use_case: "physics_decision",
        auth_mode: "oauth",
      }),
    error => error instanceof ModelProviderConfigError && error.code === "high_stakes_requires_high_reasoning",
  )
})

test("parseModelProviderConfig accepts canonical gateway auth for OAuth gateway", () => {
  const config = parseModelProviderConfig({
    provider: "oauth_gateway",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "http://127.0.0.1:8787/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: true,
    api_key_env: "MODEL_GATEWAY_TOKEN",
    auth_mode: "gateway",
    auth_refresh_command: "model-gateway auth refresh --print",
  })

  assert.equal(config.provider, "oauth_gateway")
  assert.equal(config.authMode, "gateway")
  assert.deepEqual(toPythonModelProviderAdapter(config), {
    provider: "oauth_gateway",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "http://127.0.0.1:8787/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: true,
    api_key_env: "MODEL_GATEWAY_TOKEN",
    auth_mode: "gateway",
    auth_refresh_command: "model-gateway auth refresh --print",
  })
})

test("parseModelProviderConfig rejects legacy gateway_token auth mode", () => {
  assert.throws(
    () =>
      parseModelProviderConfig({
        provider: "oauth_gateway",
        model: "gpt-5.5",
        reasoning_effort: "high",
        base_url: "http://127.0.0.1:8787/v1",
        auth_mode: "gateway_token",
      }),
    error => error instanceof ModelProviderConfigError && error.code === "invalid_auth_mode",
  )
})

test("CredentialManager refreshes expired OAuth credentials through provider boundary", async () => {
  const provider = new FakeOAuthProvider({
    id: "oauth_gateway",
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
    await first.set("oauth_gateway", {
      access: "access:persisted",
      refresh: "refresh:persisted",
      expires: 123,
    })

    const second = new FileCredentialStore(path)
    const stored = await second.get("oauth_gateway")

    assert.equal(stored?.provider, "oauth_gateway")
    assert.equal(stored?.credentials.access, "access:persisted")
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test("FakeOAuthProvider exposes login auth info without network", async () => {
  let authInfo: OAuthAuthInfo | undefined
  const provider = new FakeOAuthProvider({
    id: "openclaw",
    name: "Openclaw Test",
    authUrl: "https://openclaw.local/auth",
    accountId: "acct-1",
    email: "agent@example.test",
  })

  const credentials = await provider.login({
    onAuth: info => {
      authInfo = info
    },
    onPrompt: async () => "",
  })

  assert.equal(authInfo?.url, "https://openclaw.local/auth")
  assert.equal(credentials.accountId, "acct-1")
  assert.equal(credentials.email, "agent@example.test")
})

test("buildGatewayAdapterResponse returns Python-compatible snake_case adapter", () => {
  const response = buildGatewayAdapterResponse({
    model_provider: {
      provider: "oauth_gateway",
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
  assert.equal(response.adapter.provider, "oauth_gateway")
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
      provider: "oauth_gateway",
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
    assert.equal(response["gateway_provider"], "oauth_gateway")
    assert.equal(response["output_text"], "gateway_echo_ready")
  } finally {
    await close(gateway)
    await close(upstream)
  }
})

test("createModelGatewayServer reports missing credentials as hard gateway blocker", async () => {
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "oauth_gateway",
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
