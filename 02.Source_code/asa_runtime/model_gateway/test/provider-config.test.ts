import assert from "node:assert/strict"
import test from "node:test"
import { ModelProviderConfigError, parseModelProviderConfig, toPythonModelProviderAdapter } from "../src/index.js"

test("parseModelProviderConfig accepts explicit direct OpenAI provider config", () => {
  const config = parseModelProviderConfig({
    provider: "openai",
    model: "gpt-5.5",
    api_protocol: "responses",
    reasoning_effort: "high",
    thinking_mode: "enabled",
    base_url: "https://api.openai.com/v1/",
    auth_mode: "api_key",
    tool_choice_support: true,
    provider_session_support: true,
    credential_source: "api_key_env",
  })

  assert.equal(config.provider, "openai")
  assert.equal(config.providerId, "openai")
  assert.equal(config.apiProtocol, "responses")
  assert.equal(config.thinkingMode, "enabled")
  assert.equal(config.baseUrl, "https://api.openai.com/v1")
  assert.equal(config.toolChoiceSupport, true)
  assert.equal(config.providerSessionSupport, true)
  assert.equal(config.credentialSource, "api_key_env")
  assert.deepEqual(toPythonModelProviderAdapter(config), {
    provider: "openai",
    provider_id: "openai",
    model: "gpt-5.5",
    api_protocol: "responses",
    reasoning_effort: "high",
    thinking_mode: "enabled",
    base_url: "https://api.openai.com/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: false,
    tool_choice_support: true,
    provider_session_support: true,
    api_key_env: "OPENAI_API_KEY",
    auth_mode: "api_key",
    credential_source: "api_key_env",
  })
})

test("parseModelProviderConfig defaults ChatGPT OAuth without gateway credential names", () => {
  const config = parseModelProviderConfig({
    provider: "openai-codex",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "https://chatgpt.com/backend-api",
  })

  assert.equal(config.provider, "openai-codex")
  assert.equal(config.authMode, "oauth")
  assert.equal(config.apiKeyEnv, "ASA_OPENAI_CODEX_TOKEN")
  assert.equal(config.credentialSource, "oauth_token")
})

test("parseModelProviderConfig accepts xhigh and max reasoning at the config layer", () => {
  const xhigh = parseModelProviderConfig({
    provider_id: "local_gateway",
    model: "gpt-5.5",
    reasoning_effort: "xhigh",
    base_url: "http://127.0.0.1:8787/v1",
    auth_mode: "gateway",
  })
  const max = parseModelProviderConfig({
    provider_id: "openai",
    model: "gpt-5.5",
    reasoning_effort: "max",
    base_url: "https://api.openai.com/v1",
    auth_mode: "api_key",
  })

  assert.equal(xhigh.provider, "local_gateway")
  assert.equal(xhigh.providerId, "local_gateway")
  assert.equal(xhigh.reasoningEffort, "xhigh")
  assert.equal(max.reasoningEffort, "max")
})

test("parseModelProviderConfig rejects low reasoning for physics decisions", () => {
  assert.throws(
    () =>
      parseModelProviderConfig({
        provider: "local_gateway",
        model: "gpt-5.3-codex-spark",
        reasoning_effort: "low",
        base_url: "http://127.0.0.1:8787/v1",
        use_case: "physics_decision",
        auth_mode: "gateway",
      }),
    error => error instanceof ModelProviderConfigError && error.code === "high_stakes_requires_high_reasoning",
  )
})

test("parseModelProviderConfig accepts canonical gateway auth for local gateway", () => {
  const config = parseModelProviderConfig({
    provider: "local_gateway",
    model: "gpt-5.5",
    reasoning_effort: "high",
    base_url: "http://127.0.0.1:8787/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: true,
    api_key_env: "RUNTIME_GATEWAY_TOKEN",
    auth_mode: "gateway",
    auth_refresh_command: "model-gateway auth refresh --print",
  })

  assert.equal(config.provider, "local_gateway")
  assert.equal(config.authMode, "gateway")
  assert.deepEqual(toPythonModelProviderAdapter(config), {
    provider: "local_gateway",
    provider_id: "local_gateway",
    model: "gpt-5.5",
    api_protocol: "openai_compatible",
    reasoning_effort: "high",
    thinking_mode: "auto",
    base_url: "http://127.0.0.1:8787/v1",
    use_case: "primary_control",
    structured_outputs: true,
    streaming: true,
    tool_choice_support: true,
    provider_session_support: true,
    api_key_env: "RUNTIME_GATEWAY_TOKEN",
    auth_mode: "gateway",
    credential_source: "gateway_token",
    auth_refresh_command: "model-gateway auth refresh --print",
  })
})

test("parseModelProviderConfig rejects legacy gateway_token auth mode", () => {
  assert.throws(
    () =>
      parseModelProviderConfig({
        provider: "local_gateway",
        model: "gpt-5.5",
        reasoning_effort: "high",
        base_url: "http://127.0.0.1:8787/v1",
        auth_mode: "gateway_token",
      }),
    error => error instanceof ModelProviderConfigError && error.code === "invalid_auth_mode",
  )
})
