import assert from "node:assert/strict"
import { createServer, type Server } from "node:http"
import type { AddressInfo } from "node:net"
import test from "node:test"
import { createModelGatewayServer } from "../src/index.js"

test("createModelGatewayServer forwards openai-codex with ChatGPT headers and normalizes SSE", async () => {
  let upstreamHeaders: Record<string, string | string[] | undefined> = {}
  let upstreamBody: Record<string, unknown> = {}
  const upstream = createServer((request, response) => {
    upstreamHeaders = request.headers
    const chunks: Buffer[] = []
    request.on("data", chunk => chunks.push(chunk as Buffer))
    request.on("end", () => {
      upstreamBody = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>
      response.statusCode = 200
      response.setHeader("content-type", "text/event-stream")
      response.end(
        [
          'data: {"type":"response.output_item.done","item":{"type":"function_call","name":"artifact_write","arguments":"{\\"relative_path\\":\\"provider/evidence.txt\\",\\"content\\":\\"ok\\"}"}}',
          "",
          "data: [DONE]",
          "",
        ].join("\n"),
      )
    })
  })
  await listen(upstream)
  const token = fakeCodexJwt("acct_asa_test")
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "openai-codex",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "gateway",
    },
    upstreamBaseUrl: `${rootUrl(upstream)}/codex`,
    apiKey: token,
    requestIdFactory: () => "gw-codex-1",
  })
  await listen(gateway)

  try {
    const response = await postJson(`${rootUrl(gateway)}/v1/responses`, {
      model: "gpt-5.5",
      input: [{ role: "user", content: "write evidence" }],
      tools: [
        {
          type: "function",
          name: "artifact_write",
          description: "write evidence",
          parameters: { type: "object", properties: {}, additionalProperties: true },
        },
      ],
      reasoning: { effort: "high" },
    })
    const output = response["output"]

    assert.equal(upstreamHeaders["chatgpt-account-id"], "acct_asa_test")
    assert.equal(upstreamHeaders["openai-beta"], "responses=experimental")
    assert.equal(upstreamHeaders["originator"], "asa")
    assert.equal(upstreamBody["stream"], true)
    assert.equal(upstreamBody["store"], false)
    assert.equal(Array.isArray(upstreamBody["tools"]), true)
    assert.equal(response["gateway_request_id"], "gw-codex-1")
    assert.equal(response["gateway_provider"], "openai-codex")
    assert.equal(Array.isArray(output), true)
    assert.equal((output as Record<string, unknown>[])[0]?.["name"], "artifact_write")
  } finally {
    await close(gateway)
    await close(upstream)
  }
})

test("createModelGatewayServer forwards openai-codex to the default upstream without explicit base", async () => {
  let target = ""
  const token = fakeCodexJwt("acct_asa_default")
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "openai-codex",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "gateway",
    },
    apiKey: token,
    requestIdFactory: () => "gw-codex-default",
    fetchImpl: async input => {
      target = String(input)
      return new Response(
        'data: {"type":"response.output_text.delta","delta":"ok"}\n\ndata: [DONE]\n\n',
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    },
  })
  await listen(gateway)

  try {
    const response = await postJson(`${rootUrl(gateway)}/v1/responses`, { input: "default upstream" })

    assert.equal(target, "https://chatgpt.com/backend-api/codex/responses")
    assert.equal(response["gateway_provider"], "openai-codex")
    assert.equal(response["output_text"], "ok")
  } finally {
    await close(gateway)
  }
})

async function listen(server: Server): Promise<void> {
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
}

async function close(server: Server): Promise<void> {
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

function rootUrl(server: Server): string {
  const address = server.address() as AddressInfo
  return `http://127.0.0.1:${address.port}`
}

function fakeCodexJwt(accountId: string): string {
  const header = Buffer.from(JSON.stringify({ alg: "none" })).toString("base64url")
  const payload = Buffer.from(
    JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: accountId } }),
  ).toString("base64url")
  return `${header}.${payload}.sig`
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
