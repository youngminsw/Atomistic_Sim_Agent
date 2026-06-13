import assert from "node:assert/strict"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import type { AddressInfo } from "node:net"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"
import { createModelGatewayServer } from "../src/index.js"
import { runModelGatewayCli } from "../src/cli/commands.js"

test("CLI auth login, status, refresh, and API smoke use one credential store", async () => {
  const root = await mkdtemp(join(tmpdir(), "atomistic-cli-auth-"))
  const credentialStore = join(root, "credentials.json")
  const providerConfigPath = join(root, "provider.json")
  const smokeOutputPath = join(root, "smoke.json")
  const gateway = createModelGatewayServer({
    modelProvider: {
      provider: "oauth_gateway",
      model: "gpt-5.5",
      reasoning_effort: "high",
      base_url: "http://127.0.0.1:1/v1",
      auth_mode: "gateway",
    },
    requestIdFactory: () => "cli-gw-1",
  })
  await listen(gateway)

  try {
    const gatewayUrl = rootUrl(gateway)
    await writeFile(
      providerConfigPath,
      `${JSON.stringify({
        provider: "oauth_gateway",
        model: "gpt-5.5",
        reasoning_effort: "high",
        base_url: `${gatewayUrl}/v1`,
        auth_mode: "gateway",
      })}\n`,
    )

    const login = await runModelGatewayCli([
      "auth",
      "login",
      "--provider",
      "oauth_gateway",
      "--credential-store",
      credentialStore,
      "--access-token",
      "cli-secret-token",
      "--refresh-token",
      "cli-refresh-token",
    ])
    const status = await runModelGatewayCli([
      "auth",
      "status",
      "--provider",
      "oauth_gateway",
      "--credential-store",
      credentialStore,
      "--json",
    ])
    const refresh = await runModelGatewayCli([
      "auth",
      "refresh",
      "--provider",
      "oauth_gateway",
      "--credential-store",
      credentialStore,
      "--print",
    ])
    const smoke = await runModelGatewayCli([
      "api",
      "smoke",
      "--provider-config",
      providerConfigPath,
      "--credential-store",
      credentialStore,
      "--out",
      smokeOutputPath,
      "--input",
      "route this simulation planning request",
    ])
    const smokePayload = jsonRecord(await readFile(smokeOutputPath, "utf8"))

    assert.equal(login.code, 0)
    assert.equal(status.code, 0)
    assert.equal(refresh.code, 0)
    assert.equal(smoke.code, 0)
    assert.match(login.stdout, /login_ok=true/)
    assert.match(status.stdout, /"logged_in":true/)
    assert.doesNotMatch(status.stdout, /cli-secret-token/)
    assert.equal(refresh.stdout.trim(), "cli-secret-token")
    assert.equal(smokePayload["ok"], true)
    assert.equal(smokePayload["gateway_request_id"], "cli-gw-1")
  } finally {
    await close(gateway)
    await rm(root, { recursive: true, force: true })
  }
})

async function listen(server: ReturnType<typeof createModelGatewayServer>): Promise<void> {
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
}

async function close(server: ReturnType<typeof createModelGatewayServer>): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.close(error => {
      if (error !== undefined) {
        reject(error)
        return
      }
      resolve()
    })
  })
}

function rootUrl(server: ReturnType<typeof createModelGatewayServer>): string {
  const address = server.address()
  if (!isAddressInfo(address)) {
    throw new TypeError("expected_tcp_server_address")
  }
  return `http://127.0.0.1:${address.port}`
}

function isAddressInfo(value: AddressInfo | string | null): value is AddressInfo {
  return typeof value === "object" && value !== null && typeof value.port === "number"
}

function jsonRecord(raw: string): Record<string, unknown> {
  const value: unknown = JSON.parse(raw)
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("json_object_required")
  }
  return Object.fromEntries(Object.entries(value))
}
