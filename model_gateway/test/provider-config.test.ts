import assert from "node:assert/strict"
import test from "node:test"
import { ModelProviderConfigError, parseModelProviderConfig, toPythonModelProviderAdapter } from "../src/index.js"

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
