import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http"
import { randomUUID } from "node:crypto"
import {
  parseModelProviderConfig,
  type ModelProviderConfig,
  type PythonModelProviderAdapter,
  toPythonModelProviderAdapter,
} from "../provider/config.js"

export type ModelGatewayServerOptions = {
  readonly modelProvider: unknown
  readonly upstreamBaseUrl?: string
  readonly apiKey?: string
  readonly requestIdFactory?: () => string
  readonly fetchImpl?: typeof fetch
}

type ErrorCode =
  | "gateway_not_found"
  | "gateway_method_not_allowed"
  | "gateway_bad_json"
  | "missing_gateway_credentials"
  | "upstream_endpoint_error"

type HandlerContext = {
  readonly config: ModelProviderConfig
  readonly adapter: PythonModelProviderAdapter
  readonly requestId: string
  readonly options: ModelGatewayServerOptions
}

export function createModelGatewayServer(options: ModelGatewayServerOptions): Server {
  const config = parseModelProviderConfig(options.modelProvider)
  const adapter = toPythonModelProviderAdapter(config)
  return createServer((request, response) => {
    const context = {
      config,
      adapter,
      requestId: options.requestIdFactory?.() ?? `gw_${randomUUID()}`,
      options,
    }
    void routeRequest(context, request, response)
  })
}

async function routeRequest(context: HandlerContext, request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? "/", "http://127.0.0.1")
  response.setHeader("x-request-id", context.requestId)
  if (request.method === "GET" && url.pathname === "/healthz") {
    writeJson(response, 200, {
      ok: true,
      service: "atomistic_model_gateway",
      provider: context.config.provider,
      model: context.config.model,
      auth_mode: context.config.authMode,
      gateway_request_id: context.requestId,
    })
    return
  }
  if (request.method === "GET" && url.pathname === "/v1/models") {
    writeJson(response, 200, {
      object: "list",
      gateway_request_id: context.requestId,
      data: [{ id: context.config.model, object: "model", owned_by: context.config.provider }],
    })
    return
  }
  if (request.method === "POST" && url.pathname === "/v1/responses") {
    await handleResponses(context, request, response)
    return
  }
  writeGatewayError(response, 404, "gateway_not_found", context.requestId)
}

async function handleResponses(context: HandlerContext, request: IncomingMessage, response: ServerResponse): Promise<void> {
  const credential = context.options.apiKey ?? bearerToken(request)
  if (context.config.authMode !== "none" && credential === undefined) {
    writeGatewayError(response, 401, "missing_gateway_credentials", context.requestId)
    return
  }
  const bodyText = await readBody(request)
  const body = parseJson(bodyText)
  if (body === undefined) {
    writeGatewayError(response, 400, "gateway_bad_json", context.requestId)
    return
  }
  if (context.options.upstreamBaseUrl !== undefined) {
    await forwardToUpstream(context, response, bodyText, credential)
    return
  }
  writeJson(response, 200, {
    id: `resp_${context.requestId}`,
    object: "response",
    model: context.config.model,
    gateway_request_id: context.requestId,
    adapter: context.adapter,
    output_text: "gateway_echo_ready",
    request: body,
  })
}

async function forwardToUpstream(
  context: HandlerContext,
  response: ServerResponse,
  bodyText: string,
  credential: string | undefined,
): Promise<void> {
  const fetchImpl = context.options.fetchImpl ?? fetch
  const target = `${context.options.upstreamBaseUrl?.replace(/\/+$/, "")}/responses`
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "x-gateway-request-id": context.requestId,
  }
  if (credential !== undefined) {
    headers["authorization"] = `Bearer ${credential}`
  }
  try {
    const upstream = await fetchImpl(target, { method: "POST", headers, body: bodyText })
    const upstreamText = await upstream.text()
    const upstreamBody = parseJson(upstreamText)
    if (upstreamBody !== undefined && typeof upstreamBody === "object" && !Array.isArray(upstreamBody)) {
      writeJson(response, upstream.status, {
        ...upstreamBody,
        gateway_request_id: context.requestId,
        gateway_provider: context.config.provider,
        gateway_model: context.config.model,
      })
      return
    }
    writeJson(response, upstream.status, {
      gateway_request_id: context.requestId,
      gateway_provider: context.config.provider,
      gateway_model: context.config.model,
      upstream_text: upstreamText,
    })
  } catch (error) {
    writeGatewayError(response, 502, "upstream_endpoint_error", context.requestId, error)
  }
}

function bearerToken(request: IncomingMessage): string | undefined {
  const value = request.headers.authorization
  if (typeof value !== "string") {
    return undefined
  }
  const match = /^Bearer\s+(.+)$/iu.exec(value)
  return match?.[1]
}

async function readBody(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of request) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk)
  }
  return Buffer.concat(chunks).toString("utf8")
}

function parseJson(value: string): unknown | undefined {
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

function writeGatewayError(
  response: ServerResponse,
  status: number,
  code: ErrorCode,
  requestId: string,
  cause?: unknown,
): void {
  writeJson(response, status, {
    error: {
      code,
      message: code,
      cause: cause instanceof Error ? cause.message : undefined,
    },
    gateway_request_id: requestId,
  })
}

function writeJson(response: ServerResponse, status: number, payload: unknown): void {
  response.statusCode = status
  response.setHeader("content-type", "application/json")
  response.end(`${JSON.stringify(payload)}\n`)
}
