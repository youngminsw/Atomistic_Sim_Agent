import {
  defaultApiKeyEnv,
  defaultApiProtocol,
  defaultAuthMode,
  defaultCredentialSource,
  ModelProviderConfigError,
  parseApiProtocol,
  parseAuthMode,
  parseCredentialSource,
  parseReasoningEffort,
  parseThinkingMode,
  parseUseCase,
  type ApiProtocol,
  type AuthMode,
  type CredentialSource,
  type ModelUseCase,
  type ReasoningEffort,
  type ThinkingMode,
} from "./config-values.js"

export {
  ModelProviderConfigError,
  type ApiProtocol,
  type AuthMode,
  type CredentialSource,
  type ModelUseCase,
  type ReasoningEffort,
  type ThinkingMode,
} from "./config-values.js"

export type ModelProviderConfig = {
  readonly provider: string
  readonly providerId: string
  readonly model: string
  readonly apiProtocol: ApiProtocol
  readonly reasoningEffort: ReasoningEffort
  readonly thinkingMode: ThinkingMode
  readonly baseUrl: string
  readonly useCase: ModelUseCase
  readonly structuredOutputs: boolean
  readonly streaming: boolean
  readonly toolChoiceSupport: boolean
  readonly providerSessionSupport: boolean
  readonly apiKeyEnv: string
  readonly authMode: AuthMode
  readonly credentialSource: CredentialSource
  readonly authRefreshCommand?: string
}

export type PythonModelProviderAdapter = {
  readonly provider: string
  readonly provider_id: string
  readonly model: string
  readonly api_protocol: ApiProtocol
  readonly reasoning_effort: ReasoningEffort
  readonly thinking_mode: ThinkingMode
  readonly base_url: string
  readonly use_case: ModelUseCase
  readonly structured_outputs: boolean
  readonly streaming: boolean
  readonly tool_choice_support: boolean
  readonly provider_session_support: boolean
  readonly api_key_env: string
  readonly auth_mode: AuthMode
  readonly credential_source: CredentialSource
  readonly auth_refresh_command?: string
}

const DEFAULT_MODEL = "gpt-5.5"
const DEFAULT_REASONING: ReasoningEffort = "high"
const DEFAULT_USE_CASE: ModelUseCase = "primary_control"

export function parseModelProviderConfig(value: unknown): ModelProviderConfig {
  const mapping = objectRecord(value, "model_provider")
  const provider = requiredStringAlias(mapping, "provider_id", "provider").trim().toLowerCase()
  if (provider.length === 0) {
    throw new ModelProviderConfigError("provider_required", "provider must be a non-empty string")
  }
  const authMode = parseAuthMode(optionalStringAlias(mapping, "auth_mode", "authMode", defaultAuthMode(provider)))
  const baseConfig = {
    provider,
    providerId: provider,
    model: optionalString(mapping, "model", DEFAULT_MODEL),
    apiProtocol: parseApiProtocol(optionalStringAlias(mapping, "api_protocol", "apiProtocol", defaultApiProtocol(provider))),
    reasoningEffort: parseReasoningEffort(optionalStringAlias(mapping, "reasoning_effort", "reasoningEffort", DEFAULT_REASONING)),
    thinkingMode: parseThinkingMode(optionalStringAlias(mapping, "thinking_mode", "thinkingMode", "auto")),
    baseUrl: normalizeBaseUrl(requiredStringAlias(mapping, "base_url", "baseUrl")),
    useCase: parseUseCase(optionalStringAlias(mapping, "use_case", "useCase", DEFAULT_USE_CASE)),
    structuredOutputs: optionalBooleanAlias(mapping, "structured_outputs", "structuredOutputs", true),
    streaming: optionalBoolean(mapping, "streaming", false),
    toolChoiceSupport: optionalBooleanAlias(mapping, "tool_choice_support", "toolChoiceSupport", true),
    providerSessionSupport: optionalBooleanAlias(mapping, "provider_session_support", "providerSessionSupport", true),
    apiKeyEnv: optionalStringAlias(mapping, "api_key_env", "apiKeyEnv", defaultApiKeyEnv(provider)),
    authMode,
    credentialSource: parseCredentialSource(
      optionalStringAlias(mapping, "credential_source", "credentialSource", defaultCredentialSource(authMode)),
    ),
  }
  const authRefreshCommand = optionalStringOrUndefinedAlias(mapping, "auth_refresh_command", "authRefreshCommand")
  const config = authRefreshCommand === undefined ? baseConfig : { ...baseConfig, authRefreshCommand }
  enforceModelProviderPolicy(config)
  return config
}

export function toPythonModelProviderAdapter(config: ModelProviderConfig): PythonModelProviderAdapter {
  const adapter = {
    provider: config.provider,
    provider_id: config.providerId,
    model: config.model,
    api_protocol: config.apiProtocol,
    reasoning_effort: config.reasoningEffort,
    thinking_mode: config.thinkingMode,
    base_url: config.baseUrl,
    use_case: config.useCase,
    structured_outputs: config.structuredOutputs,
    streaming: config.streaming,
    tool_choice_support: config.toolChoiceSupport,
    provider_session_support: config.providerSessionSupport,
    api_key_env: config.apiKeyEnv,
    auth_mode: config.authMode,
    credential_source: config.credentialSource,
  }
  if (config.authRefreshCommand === undefined) {
    return adapter
  }
  return { ...adapter, auth_refresh_command: config.authRefreshCommand }
}

export function enforceModelProviderPolicy(config: ModelProviderConfig): void {
  const highStakes = new Set<ModelUseCase>(["primary_control", "physics_decision", "final_run_approval"])
  const highReasoning = new Set<ReasoningEffort>(["high", "xhigh", "max"])
  if (highStakes.has(config.useCase) && !highReasoning.has(config.reasoningEffort)) {
    throw new ModelProviderConfigError(
      "high_stakes_requires_high_reasoning",
      "primary control, physics decisions, and final approvals require high reasoning",
    )
  }
}

function normalizeBaseUrl(value: string): string {
  let url: URL
  try {
    url = new URL(value)
  } catch (error) {
    throw new ModelProviderConfigError("invalid_base_url", "base_url must be an absolute HTTP(S) URL")
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new ModelProviderConfigError("invalid_base_url", "base_url must not include auth, query, or fragment parts")
  }
  url.pathname = url.pathname.replace(/\/+$/, "")
  return url.toString().replace(/\/$/, "")
}

function objectRecord(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ModelProviderConfigError("object_required", `${field} must be an object`)
  }
  return value as Record<string, unknown>
}

function requiredString(mapping: Record<string, unknown>, field: string): string {
  const value = mapping[field]
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${field} must be a non-empty string`)
  }
  return value
}

function requiredStringAlias(mapping: Record<string, unknown>, snakeField: string, camelField: string): string {
  const value = mapping[snakeField] ?? mapping[camelField]
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${snakeField} must be a non-empty string`)
  }
  return value
}

function optionalString(mapping: Record<string, unknown>, field: string, fallback: string): string {
  const value = mapping[field]
  if (value === undefined) {
    return fallback
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${field} must be a non-empty string`)
  }
  return value
}

function optionalStringAlias(
  mapping: Record<string, unknown>,
  snakeField: string,
  camelField: string,
  fallback: string,
): string {
  const value = mapping[snakeField] ?? mapping[camelField]
  if (value === undefined) {
    return fallback
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${snakeField} must be a non-empty string`)
  }
  return value
}

function optionalStringOrUndefined(mapping: Record<string, unknown>, field: string): string | undefined {
  const value = mapping[field]
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${field} must be a non-empty string`)
  }
  return value
}

function optionalStringOrUndefinedAlias(
  mapping: Record<string, unknown>,
  snakeField: string,
  camelField: string,
): string | undefined {
  const value = mapping[snakeField] ?? mapping[camelField]
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelProviderConfigError("string_required", `${snakeField} must be a non-empty string`)
  }
  return value
}

function optionalBoolean(mapping: Record<string, unknown>, field: string, fallback: boolean): boolean {
  const value = mapping[field]
  if (value === undefined) {
    return fallback
  }
  if (typeof value !== "boolean") {
    throw new ModelProviderConfigError("boolean_required", `${field} must be boolean`)
  }
  return value
}

function optionalBooleanAlias(
  mapping: Record<string, unknown>,
  snakeField: string,
  camelField: string,
  fallback: boolean,
): boolean {
  const value = mapping[snakeField] ?? mapping[camelField]
  if (value === undefined) {
    return fallback
  }
  if (typeof value !== "boolean") {
    throw new ModelProviderConfigError("boolean_required", `${snakeField} must be boolean`)
  }
  return value
}
