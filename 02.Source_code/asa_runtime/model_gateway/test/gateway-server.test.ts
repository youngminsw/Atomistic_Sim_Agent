import assert from "node:assert/strict"
import type { AddressInfo } from "node:net"
import { createServer, type Server } from "node:http"
import test from "node:test"
import { createModelGatewayServer } from "../src/index.js"

test("echo-redaction returns bounded metadata without prompt token messages or tool arguments", async () => {
  // Given: echo mode receives sensitive request fields.
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "none",
    },
    requestIdFactory: () => "gw-echo-redaction",
  })
  await listen(gateway)
  const secretPrompt = "prompt-secret-task-13"
  const secretToken = "token-secret-task-13"
  const toolArgument = "tool-argument-secret-task-13"

  try {
    // When: a response is requested without upstream forwarding.
    const response = await fetch(`${rootUrl(gateway)}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${secretToken}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-5.5",
        input: secretPrompt,
        messages: [{ role: "user", content: secretPrompt }],
        tools: [{ name: "artifact_write", arguments: { content: toolArgument } }],
      }),
    })
    const bodyText = await response.text()
    const body = JSON.parse(bodyText) as Record<string, unknown>

    // Then: only stable diagnostics and bounded metadata are returned.
    assert.equal(response.status, 200)
    assert.equal(body["gateway_request_id"], "gw-echo-redaction")
    assert.equal(body["model"], "gpt-5.5")
    assert.equal(body["output_text"], "gateway_echo_ready")
    assert.ok("request_metadata" in body)
    assert.ok(!("request" in body))
    assert.doesNotMatch(bodyText, /prompt-secret-task-13/)
    assert.doesNotMatch(bodyText, /token-secret-task-13/)
    assert.doesNotMatch(bodyText, /tool-argument-secret-task-13/)
    assert.doesNotMatch(bodyText, /"messages"/)
  } finally {
    await close(gateway)
  }
})

test("upstream-redaction omits raw upstream body from error responses", async () => {
  // Given: an upstream endpoint returns a non-JSON error body with a secret tail.
  const upstreamSecret = "upstream-raw-secret-task-13"
  const upstream = createServer((_request, response) => {
    response.statusCode = 503
    response.setHeader("content-type", "text/plain")
    response.end(`provider failed ${upstreamSecret}`)
  })
  await listen(upstream)
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "local_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:8787/v1",
      auth_mode: "none",
    },
    upstreamBaseUrl: `${rootUrl(upstream)}/v1`,
    requestIdFactory: () => "gw-upstream-redaction",
  })
  await listen(gateway)

  try {
    // When: the gateway forwards to the failing upstream.
    const response = await fetch(`${rootUrl(gateway)}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-5.5", input: "hello" }),
    })
    const bodyText = await response.text()
    const body = JSON.parse(bodyText) as Record<string, unknown>

    // Then: status and request diagnostics remain, but the raw upstream body is redacted.
    assert.equal(response.status, 503)
    assert.equal(body["gateway_request_id"], "gw-upstream-redaction")
    assert.equal(body["gateway_provider"], "local_gateway")
    assert.ok("upstream_metadata" in body)
    assert.ok(!("upstream_text" in body))
    assert.doesNotMatch(bodyText, /upstream-raw-secret-task-13/)
  } finally {
    await close(gateway)
    await close(upstream)
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
