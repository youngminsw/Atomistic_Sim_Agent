import { readFile, writeFile } from "node:fs/promises"
import { createModelGatewayServer } from "../gateway/server.js"
import { parseModelProviderConfig, type ModelProviderConfig } from "../provider/config.js"
import { FileCredentialStore, type StoredCredential } from "../storage/credential-store.js"
import { CliArgsError, hasFlag, optionalOption, parseArgs, requiredOption } from "./args.js"

const DEFAULT_STORE = ".atomistic-sim-agent/model-gateway-credentials.json"
const DEFAULT_EXPIRES_IN_MS = 3_600_000

export type CliRunResult = {
  readonly code: number
  readonly stdout: string
  readonly stderr: string
}

export type CliRuntime = {
  readonly env?: Readonly<Record<string, string | undefined>>
}

export async function runModelGatewayCli(argv: readonly string[], runtime: CliRuntime = {}): Promise<CliRunResult> {
  try {
    return await runParsedCommand(parseArgs(argv), runtime)
  } catch (error) {
    if (error instanceof CliArgsError) {
      return fail(error.code)
    }
    if (error instanceof Error) {
      return fail(error.message)
    }
    return fail("unknown_cli_error")
  }
}

async function runParsedCommand(args: ReturnType<typeof parseArgs>, runtime: CliRuntime): Promise<CliRunResult> {
  const command = args.positionals[0]
  const subcommand = args.positionals[1]
  if (command === "auth" && subcommand === "login") {
    return await authLogin(args, runtime)
  }
  if (command === "auth" && subcommand === "status") {
    return await authStatus(args)
  }
  if (command === "auth" && subcommand === "refresh") {
    return await authRefresh(args)
  }
  if (command === "api" && subcommand === "smoke") {
    return await apiSmoke(args)
  }
  if (command === "serve") {
    return await serveGateway(args)
  }
  return fail(helpText())
}

async function authLogin(args: ReturnType<typeof parseArgs>, runtime: CliRuntime): Promise<CliRunResult> {
  const provider = requiredOption(args, "provider")
  const token = loginToken(args, runtime)
  if (token === undefined) {
    return fail("missing_access_token")
  }
  const refresh = optionalOption(args, "refresh-token") ?? token
  const expiresInMs = numberOption(args, "expires-in-ms", DEFAULT_EXPIRES_IN_MS)
  const store = new FileCredentialStore(storePath(args))
  const stored = await store.set(provider, {
    access: token,
    refresh,
    expires: Date.now() + expiresInMs,
  })
  return ok(`login_ok=true\nprovider=${stored.provider}\nexpires=${stored.credentials.expires}\n`)
}

async function authStatus(args: ReturnType<typeof parseArgs>): Promise<CliRunResult> {
  const provider = requiredOption(args, "provider")
  const stored = await new FileCredentialStore(storePath(args)).get(provider)
  const payload = {
    provider,
    logged_in: stored !== undefined,
    expires: stored?.credentials.expires ?? null,
    updated_at_ms: stored?.updatedAtMs ?? null,
  }
  if (hasFlag(args, "json")) {
    return ok(`${JSON.stringify(payload)}\n`)
  }
  return ok(`provider=${provider}\nlogged_in=${String(stored !== undefined)}\n`)
}

async function authRefresh(args: ReturnType<typeof parseArgs>): Promise<CliRunResult> {
  const provider = requiredOption(args, "provider")
  const stored = await requireStoredCredential(args, provider)
  if (hasFlag(args, "print")) {
    return ok(`${stored.credentials.access}\n`)
  }
  return ok(`refresh_ok=true\nprovider=${provider}\nexpires=${stored.credentials.expires}\n`)
}

async function apiSmoke(args: ReturnType<typeof parseArgs>): Promise<CliRunResult> {
  const config = await providerConfigFromFile(requiredOption(args, "provider-config"))
  const stored = await credentialForConfig(args, config)
  const token = stored?.credentials.access
  const health = await getJson(url(config.baseUrl, "/healthz"), token)
  const models = await getJson(url(config.baseUrl, "/v1/models"), token)
  const response = await postJson(url(config.baseUrl, "/v1/responses"), token, {
    model: config.model,
    input: optionalOption(args, "input") ?? "model gateway smoke",
  })
  const payload = {
    ok: health["ok"] === true,
    provider: config.provider,
    model: config.model,
    models_count: Array.isArray(models["data"]) ? models["data"].length : 0,
    gateway_request_id: response["gateway_request_id"] ?? null,
    output_text: response["output_text"] ?? "",
  }
  const out = optionalOption(args, "out")
  if (out !== undefined) {
    await writeFile(out, `${JSON.stringify(payload, null, 2)}\n`, "utf8")
  }
  return ok(`${JSON.stringify(payload)}\n`)
}

async function serveGateway(args: ReturnType<typeof parseArgs>): Promise<CliRunResult> {
  const config = await providerConfigFromFile(requiredOption(args, "provider-config"))
  const stored = await credentialForConfig(args, config)
  const upstreamBaseUrl = optionalOption(args, "upstream-base-url")
  const baseOptions = {
    modelProvider: config,
  }
  const withUpstream = upstreamBaseUrl === undefined ? baseOptions : { ...baseOptions, upstreamBaseUrl }
  const options = stored === undefined ? withUpstream : { ...withUpstream, apiKey: stored.credentials.access }
  const server = createModelGatewayServer({
    ...options,
  })
  const host = optionalOption(args, "host") ?? "127.0.0.1"
  const port = numberOption(args, "port", 8787)
  await new Promise<void>(resolve => server.listen(port, host, resolve))
  return ok(`gateway_listening=http://${host}:${port}\n`)
}

function loginToken(args: ReturnType<typeof parseArgs>, runtime: CliRuntime): string | undefined {
  const direct = optionalOption(args, "access-token")
  if (direct !== undefined) {
    return direct
  }
  const envName = optionalOption(args, "access-token-env")
  if (envName === undefined) {
    return undefined
  }
  return runtime.env?.[envName] ?? process.env[envName]
}

async function providerConfigFromFile(path: string): Promise<ModelProviderConfig> {
  const value: unknown = JSON.parse(await readFile(path, "utf8"))
  return parseModelProviderConfig(value)
}

async function credentialForConfig(
  args: ReturnType<typeof parseArgs>,
  config: ModelProviderConfig,
): Promise<StoredCredential | undefined> {
  if (config.authMode === "none") {
    return undefined
  }
  return await requireStoredCredential(args, config.provider)
}

async function requireStoredCredential(args: ReturnType<typeof parseArgs>, provider: string): Promise<StoredCredential> {
  const stored = await new FileCredentialStore(storePath(args)).get(provider)
  if (stored === undefined) {
    throw new CliArgsError(`missing_credentials:${provider}`)
  }
  return stored
}

function storePath(args: ReturnType<typeof parseArgs>): string {
  return optionalOption(args, "credential-store") ?? `${process.env["HOME"] ?? "."}/${DEFAULT_STORE}`
}

function numberOption(args: ReturnType<typeof parseArgs>, key: string, fallback: number): number {
  const raw = optionalOption(args, key)
  if (raw === undefined) {
    return fallback
  }
  const value = Number(raw)
  if (!Number.isFinite(value) || value < 0) {
    throw new CliArgsError(`invalid_number:${key}`)
  }
  return value
}

async function getJson(rawUrl: string, token: string | undefined): Promise<Record<string, unknown>> {
  return await requestJson(rawUrl, "GET", token)
}

async function postJson(rawUrl: string, token: string | undefined, payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return await requestJson(rawUrl, "POST", token, JSON.stringify(payload))
}

async function requestJson(
  rawUrl: string,
  method: "GET" | "POST",
  token: string | undefined,
  body?: string,
): Promise<Record<string, unknown>> {
  const headers = token === undefined ? { "content-type": "application/json" } : {
    authorization: `Bearer ${token}`,
    "content-type": "application/json",
  }
  const init = body === undefined ? { method, headers, signal: AbortSignal.timeout(10_000) } : {
    method,
    headers,
    body,
    signal: AbortSignal.timeout(10_000),
  }
  const response = await fetch(rawUrl, init)
  const value: unknown = JSON.parse(await response.text())
  if (!response.ok) {
    throw new CliArgsError(`endpoint_http_${response.status}`)
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CliArgsError("endpoint_json_object_required")
  }
  return Object.fromEntries(Object.entries(value))
}

function url(baseUrl: string, path: string): string {
  if (path === "/healthz") {
    return `${baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "")}/healthz`
  }
  return `${baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "")}${path}`
}

function ok(stdout: string): CliRunResult {
  return { code: 0, stdout, stderr: "" }
}

function fail(stderr: string): CliRunResult {
  return { code: 1, stdout: "", stderr: `${stderr}\n` }
}

function helpText(): string {
  return "usage: model-gateway auth login|status|refresh, model-gateway api smoke, model-gateway serve"
}
