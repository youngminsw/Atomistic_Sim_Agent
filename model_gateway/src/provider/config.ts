export type ReasoningEffort = "low" | "medium" | "high"

export type ModelUseCase =
  | "primary_control"
  | "low_risk_extraction"
  | "low_risk_summarization"
  | "physics_decision"
  | "final_run_approval"

export type AuthMode = "api_key" | "oauth" | "gateway" | "none"

export type ModelProviderConfig = {
  readonly provider: string
  readonly model: string
  readonly reasoningEffort: ReasoningEffort
  readonly baseUrl: string
  readonly useCase: ModelUseCase
  readonly structuredOutputs: boolean
  readonly streaming: boolean
  readonly apiKeyEnv: string
  readonly authMode: AuthMode
  readonly authRefreshCommand?: string
}

export type PythonModelProviderAdapter = {
  readonly provider: string
  readonly model: string
  readonly reasoning_effort: ReasoningEffort
  readonly base_url: string
  readonly use_case: ModelUseCase
  readonly structured_outputs: boolean
  readonly streaming: boolean
  readonly api_key_env: string
  readonly auth_mode: AuthMode
  readonly auth_refresh_command?: string
}

export class ModelProviderConfigError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = "ModelProviderConfigError"
    this.code = code
  }
}

const DEFAULT_MODEL = "gpt-5.5"
const DEFAULT_REASONING: ReasoningEffort = "high"
const DEFAULT_USE_CASE: ModelUseCase = "primary_control"
const REASONING_EFFORTS: ReadonlySet<string> = new Set(["low", "medium", "high"])
const USE_CASES: ReadonlySet<string> = new Set([
  "primary_control",
  "low_risk_extraction",
  "low_risk_summarization",
  "physics_decision",
  "final_run_approval",
])
const AUTH_MODES: ReadonlySet<string> = new Set(["api_key", "oauth", "gateway", "none"])

export function parseModelProviderConfig(value: unknown): ModelProviderConfig {
  const mapping = objectRecord(value, "model_provider")
  const provider = requiredString(mapping, "provider").trim().toLowerCase()
  if (provider.length === 0) {
    throw new ModelProviderConfigError("provider_required", "provider must be a non-empty string")
  }
  const baseConfig = {
    provider,
    model: optionalString(mapping, "model", DEFAULT_MODEL),
    reasoningEffort: parseReasoningEffort(optionalStringAlias(mapping, "reasoning_effort", "reasoningEffort", DEFAULT_REASONING)),
    baseUrl: normalizeBaseUrl(requiredStringAlias(mapping, "base_url", "baseUrl")),
    useCase: parseUseCase(optionalStringAlias(mapping, "use_case", "useCase", DEFAULT_USE_CASE)),
    structuredOutputs: optionalBooleanAlias(mapping, "structured_outputs", "structuredOutputs", true),
    streaming: optionalBoolean(mapping, "streaming", false),
    apiKeyEnv: optionalStringAlias(mapping, "api_key_env", "apiKeyEnv", defaultApiKeyEnv(provider)),
    authMode: parseAuthMode(optionalStringAlias(mapping, "auth_mode", "authMode", defaultAuthMode(provider))),
  }
  const authRefreshCommand = optionalStringOrUndefinedAlias(mapping, "auth_refresh_command", "authRefreshCommand")
  const config = authRefreshCommand === undefined ? baseConfig : { ...baseConfig, authRefreshCommand }
  enforceModelProviderPolicy(config)
  return config
}

export function toPythonModelProviderAdapter(config: ModelProviderConfig): PythonModelProviderAdapter {
  const adapter = {
    provider: config.provider,
    model: config.model,
    reasoning_effort: config.reasoningEffort,
    base_url: config.baseUrl,
    use_case: config.useCase,
    structured_outputs: config.structuredOutputs,
    streaming: config.streaming,
    api_key_env: config.apiKeyEnv,
    auth_mode: config.authMode,
  }
  if (config.authRefreshCommand === undefined) {
    return adapter
  }
  return { ...adapter, auth_refresh_command: config.authRefreshCommand }
}

export function enforceModelProviderPolicy(config: ModelProviderConfig): void {
  const highStakes = new Set<ModelUseCase>(["primary_control", "physics_decision", "final_run_approval"])
  if (highStakes.has(config.useCase) && config.reasoningEffort !== "high") {
    throw new ModelProviderConfigError(
      "high_stakes_requires_high_reasoning",
      "primary control, physics decisions, and final approvals require high reasoning",
    )
  }
}

function parseReasoningEffort(value: string): ReasoningEffort {
  if (!REASONING_EFFORTS.has(value)) {
    throw new ModelProviderConfigError("invalid_reasoning_effort", `invalid reasoning_effort=${value}`)
  }
  return value as ReasoningEffort
}

function parseUseCase(value: string): ModelUseCase {
  if (!USE_CASES.has(value)) {
    throw new ModelProviderConfigError("invalid_use_case", `invalid use_case=${value}`)
  }
  return value as ModelUseCase
}

function parseAuthMode(value: string): AuthMode {
  if (!AUTH_MODES.has(value)) {
    throw new ModelProviderConfigError("invalid_auth_mode", `invalid auth_mode=${value}`)
  }
  return value as AuthMode
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

function defaultApiKeyEnv(provider: string): string {
  if (provider === "openai") {
    return "OPENAI_API_KEY"
  }
  if (provider === "anthropic") {
    return "ANTHROPIC_API_KEY"
  }
  if (provider === "openclaw") {
    return "OPENCLAW_OAUTH_TOKEN"
  }
  return "MODEL_GATEWAY_TOKEN"
}

function defaultAuthMode(provider: string): AuthMode {
  if (provider === "openclaw") {
    return "oauth"
  }
  if (provider === "openai" || provider === "anthropic") {
    return "api_key"
  }
  if (provider.endsWith("_gateway") || provider === "local_gateway") {
    return "gateway"
  }
  return "gateway"
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
