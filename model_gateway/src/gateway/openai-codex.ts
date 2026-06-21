import type { ModelProviderConfig } from "../provider/config.js"

export const OPENAI_CODEX_DEFAULT_UPSTREAM = "https://chatgpt.com/backend-api/codex"

const OPENAI_CODEX_ACCOUNT_CLAIM = "https://api.openai.com/auth"

export type OpenAICodexGatewayRequest = {
  readonly config: ModelProviderConfig
  readonly upstreamBaseUrl: string | undefined
  readonly credential: string
  readonly body: unknown
  readonly fetchImpl: typeof fetch | undefined
}

export type OpenAICodexGatewayResponse = {
  readonly status: number
  readonly body: Record<string, unknown>
}

export async function forwardOpenAICodexGatewayRequest(
  request: OpenAICodexGatewayRequest,
): Promise<OpenAICodexGatewayResponse> {
  const fetchImpl = request.fetchImpl ?? fetch
  const upstream = await fetchImpl(resolveOpenAICodexResponsesUrl(request.upstreamBaseUrl), {
    method: "POST",
    headers: openAICodexHeaders(request.credential),
    body: JSON.stringify(openAICodexRequestBody(request.config, request.body)),
  })
  const upstreamText = await upstream.text()
  return {
    status: upstream.status,
    body: upstreamJsonBody(upstream, upstreamText) ?? normalizeSseGatewayResponse(upstreamText),
  }
}

function resolveOpenAICodexResponsesUrl(upstreamBaseUrl: string | undefined): string {
  const raw = (upstreamBaseUrl ?? OPENAI_CODEX_DEFAULT_UPSTREAM).replace(/\/+$/, "")
  if (raw.endsWith("/codex/responses")) {
    return raw
  }
  if (raw.endsWith("/codex")) {
    return `${raw}/responses`
  }
  return `${raw}/codex/responses`
}

function openAICodexHeaders(accessToken: string): Headers {
  const headers = new Headers()
  headers.set("authorization", `Bearer ${accessToken}`)
  headers.set("content-type", "application/json")
  headers.set("accept", "text/event-stream")
  headers.set("OpenAI-Beta", "responses=experimental")
  headers.set("chatgpt-account-id", openAICodexAccountId(accessToken))
  headers.set("originator", "asa")
  headers.set("user-agent", "asa-model-gateway/0.1")
  return headers
}

function openAICodexAccountId(accessToken: string): string {
  const parts = accessToken.split(".")
  if (parts.length !== 3 || parts[1] === undefined) {
    throw new Error("openai_codex_account_id_missing")
  }
  const payload = parseJson(Buffer.from(parts[1], "base64url").toString("utf8"))
  const auth = objectValue(payload)?.[OPENAI_CODEX_ACCOUNT_CLAIM]
  const accountId = objectValue(auth)?.["chatgpt_account_id"]
  if (typeof accountId !== "string" || accountId.length === 0) {
    throw new Error("openai_codex_account_id_missing")
  }
  return accountId
}

function openAICodexRequestBody(config: ModelProviderConfig, value: unknown): Record<string, unknown> {
  const source = objectValue(value) ?? {}
  const request: Record<string, unknown> = {
    ...source,
    model: typeof source["model"] === "string" ? source["model"] : config.model,
    input: codexInput(source["input"]),
    tools: codexTools(source["tools"]),
    reasoning: codexReasoning(config, source["reasoning"]),
    text: { ...(objectValue(source["text"]) ?? {}), verbosity: "low" },
    include: codexInclude(source["include"]),
    stream: true,
    store: false,
  }
  delete request["metadata"]
  return request
}

function codexInput(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value.map(item => codexInputItem(item))
  }
  if (typeof value === "string") {
    return [codexMessage("user", value)]
  }
  return [codexMessage("user", "ASA provider smoke")]
}

function codexInputItem(value: unknown): Record<string, unknown> {
  const item = objectValue(value)
  if (item === undefined) {
    return codexMessage("user", String(value ?? ""))
  }
  if (item["type"] === "message") {
    return item
  }
  const role = typeof item["role"] === "string" ? item["role"] : "user"
  const content = item["content"]
  if (typeof content === "string") {
    return codexMessage(role, content)
  }
  return { ...item, type: "message", role }
}

function codexMessage(role: string, text: string): Record<string, unknown> {
  return { type: "message", role, content: [{ type: "input_text", text }] }
}

function codexTools(value: unknown): unknown {
  if (!Array.isArray(value)) {
    return undefined
  }
  return value.map(tool => {
    const item = objectValue(tool) ?? {}
    const name = typeof item["name"] === "string" ? item["name"] : "tool"
    return {
      type: "function",
      name,
      description: typeof item["description"] === "string" ? item["description"] : name,
      parameters: objectValue(item["parameters"]) ?? { type: "object", properties: {}, additionalProperties: true },
    }
  })
}

function codexReasoning(config: ModelProviderConfig, value: unknown): Record<string, unknown> {
  const source = objectValue(value) ?? {}
  return {
    ...source,
    effort: typeof source["effort"] === "string" ? source["effort"] : config.reasoningEffort,
    summary: "auto",
  }
}

function codexInclude(value: unknown): string[] {
  const items = Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []
  return Array.from(new Set([...items, "reasoning.encrypted_content"]))
}

function upstreamJsonBody(response: Response, text: string): Record<string, unknown> | undefined {
  const contentType = response.headers.get("content-type") ?? ""
  if (contentType.includes("text/event-stream")) {
    return undefined
  }
  return objectValue(parseJson(text))
}

function normalizeSseGatewayResponse(text: string): Record<string, unknown> {
  const events = parseSseJsonEvents(text)
  const output: unknown[] = []
  let outputText = ""
  for (const event of events) {
    const type = event["type"]
    const response = objectValue(event["response"])
    const responseOutput = response?.["output"]
    if (Array.isArray(responseOutput) && responseOutput.length > 0) {
      output.splice(0, output.length, ...responseOutput)
    }
    const item = objectValue(event["item"])
    if (type === "response.output_item.done" && item !== undefined) {
      output.push(item)
    }
    const delta = event["delta"]
    if (type === "response.output_text.delta" && typeof delta === "string") {
      outputText += delta
    }
  }
  return {
    object: "response",
    output,
    output_text: outputText || outputTextFromItems(output),
    upstream_event_count: events.length,
  }
}

function parseSseJsonEvents(text: string): Record<string, unknown>[] {
  const events: Record<string, unknown>[] = []
  for (const chunk of text.split(/\r?\n\r?\n/u)) {
    const data = chunk
      .split(/\r?\n/u)
      .filter(line => line.startsWith("data:"))
      .map(line => line.slice("data:".length).trimStart())
      .join("\n")
    if (!data || data === "[DONE]") {
      continue
    }
    const parsed = objectValue(parseJson(data))
    if (parsed !== undefined) {
      events.push(parsed)
    }
  }
  return events
}

function outputTextFromItems(items: unknown[]): string {
  const parts: string[] = []
  for (const item of items) {
    const record = objectValue(item)
    const content = record?.["content"]
    if (!Array.isArray(content)) {
      continue
    }
    for (const part of content) {
      const partRecord = objectValue(part)
      const text = partRecord?.["text"]
      if (typeof text === "string") {
        parts.push(text)
      }
    }
  }
  return parts.join("")
}

function parseJson(value: string): unknown | undefined {
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined
  }
  return value as Record<string, unknown>
}
