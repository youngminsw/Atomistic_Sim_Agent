export type ReasoningEffort = "inherit" | "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
export type ApiProtocol = "openai_compatible" | "responses" | "chat_completions" | "anthropic_messages" | "gemini"
export type ThinkingMode = "auto" | "enabled" | "disabled"
export type CredentialSource = "api_key_env" | "oauth_token" | "gateway_token" | "none"

export type ModelUseCase =
  | "primary_control"
  | "low_risk_extraction"
  | "low_risk_summarization"
  | "physics_decision"
  | "final_run_approval"

export type AuthMode = "api_key" | "oauth" | "gateway" | "none"

export class ModelProviderConfigError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = "ModelProviderConfigError"
    this.code = code
  }
}

export function parseReasoningEffort(value: string): ReasoningEffort {
  switch (value) {
    case "inherit":
    case "off":
    case "minimal":
    case "low":
    case "medium":
    case "high":
    case "xhigh":
    case "max":
      return value
    default:
      throw new ModelProviderConfigError("invalid_reasoning_effort", `invalid reasoning_effort=${value}`)
  }
}

export function parseApiProtocol(value: string): ApiProtocol {
  switch (value) {
    case "openai_compatible":
    case "responses":
    case "chat_completions":
    case "anthropic_messages":
    case "gemini":
      return value
    default:
      throw new ModelProviderConfigError("invalid_api_protocol", `invalid api_protocol=${value}`)
  }
}

export function parseThinkingMode(value: string): ThinkingMode {
  switch (value) {
    case "auto":
    case "enabled":
    case "disabled":
      return value
    default:
      throw new ModelProviderConfigError("invalid_thinking_mode", `invalid thinking_mode=${value}`)
  }
}

export function parseUseCase(value: string): ModelUseCase {
  switch (value) {
    case "primary_control":
    case "low_risk_extraction":
    case "low_risk_summarization":
    case "physics_decision":
    case "final_run_approval":
      return value
    default:
      throw new ModelProviderConfigError("invalid_use_case", `invalid use_case=${value}`)
  }
}

export function parseAuthMode(value: string): AuthMode {
  switch (value) {
    case "api_key":
    case "oauth":
    case "gateway":
    case "none":
      return value
    default:
      throw new ModelProviderConfigError("invalid_auth_mode", `invalid auth_mode=${value}`)
  }
}

export function parseCredentialSource(value: string): CredentialSource {
  switch (value) {
    case "api_key_env":
    case "oauth_token":
    case "gateway_token":
    case "none":
      return value
    default:
      throw new ModelProviderConfigError("invalid_credential_source", `invalid credential_source=${value}`)
  }
}

export function defaultApiKeyEnv(provider: string): string {
  if (provider === "openai-codex") {
    return "ASA_OPENAI_CODEX_TOKEN"
  }
  if (provider === "openai") {
    return "OPENAI_API_KEY"
  }
  if (provider === "anthropic") {
    return "ANTHROPIC_API_KEY"
  }
  if (provider === "openclaw") {
    return "OPENCLAW_OAUTH_TOKEN"
  }
  if (provider === "oauth_gateway" || provider === "anthropic_gateway") {
    return "MODEL_GATEWAY_TOKEN"
  }
  return "RUNTIME_GATEWAY_TOKEN"
}

export function defaultAuthMode(provider: string): AuthMode {
  if (provider === "openai-codex") {
    return "oauth"
  }
  if (provider === "openclaw") {
    return "oauth"
  }
  if (provider === "openai" || provider === "anthropic") {
    return "api_key"
  }
  if (provider === "local_gateway") {
    return "none"
  }
  if (provider.endsWith("_gateway")) {
    return "gateway"
  }
  return "gateway"
}

export function defaultApiProtocol(provider: string): ApiProtocol {
  if (provider === "openai" || provider === "openai-codex") {
    return "responses"
  }
  if (provider === "anthropic") {
    return "anthropic_messages"
  }
  if (provider === "google-gemini-cli" || provider === "google-antigravity") {
    return "gemini"
  }
  return "openai_compatible"
}

export function defaultCredentialSource(authMode: AuthMode): CredentialSource {
  if (authMode === "api_key") {
    return "api_key_env"
  }
  if (authMode === "oauth") {
    return "oauth_token"
  }
  if (authMode === "none") {
    return "none"
  }
  return "gateway_token"
}
