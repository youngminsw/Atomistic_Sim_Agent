import assert from "node:assert/strict"
import test from "node:test"
import { OAuthCallbackFlow, generatePKCE, parseCallbackInput } from "../src/index.js"
import type { OAuthAuthInfo, OAuthCredentials, OAuthController } from "../src/index.js"

class TestOAuthFlow extends OAuthCallbackFlow {
  constructor(ctrl: OAuthController) {
    super(ctrl, { preferredPort: 0 })
  }

  async generateAuthUrl(state: string, redirectUri: string): Promise<{ readonly url: string; readonly instructions?: string }> {
    const url = new URL("https://auth.example.test/authorize")
    url.searchParams.set("state", state)
    url.searchParams.set("redirect_uri", redirectUri)
    return { url: url.toString(), instructions: "test login" }
  }

  async exchangeToken(code: string, state: string): Promise<OAuthCredentials> {
    return {
      access: `access:${code}:${state}`,
      refresh: "refresh-token",
      expires: Date.now() + 60_000,
      accountId: "test-account",
    }
  }
}

test("parseCallbackInput returns code and state when input is a callback URL", () => {
  const parsed = parseCallbackInput("http://localhost:1455/callback?code=abc&state=xyz")
  assert.deepEqual(parsed, { code: "abc", state: "xyz" })
})

test("parseCallbackInput returns code and state when input is a query string", () => {
  const parsed = parseCallbackInput("?code=abc&state=xyz")
  assert.deepEqual(parsed, { code: "abc", state: "xyz" })
})

test("parseCallbackInput returns raw code with optional state suffix", () => {
  const parsed = parseCallbackInput("abc#xyz")
  assert.deepEqual(parsed, { code: "abc", state: "xyz" })
})

test("generatePKCE returns url-safe verifier and challenge", async () => {
  const pair = await generatePKCE()
  assert.match(pair.verifier, /^[A-Za-z0-9_-]+$/)
  assert.match(pair.challenge, /^[A-Za-z0-9_-]+$/)
  assert.notEqual(pair.verifier, pair.challenge)
})

test("OAuthCallbackFlow resolves credentials after local callback", async () => {
  let authInfo: OAuthAuthInfo | undefined
  const flow = new TestOAuthFlow({
    onAuth: info => {
      authInfo = info
      queueMicrotask(async () => {
        const authUrl = new URL(info.url)
        const redirectUri = authUrl.searchParams.get("redirect_uri")
        const state = authUrl.searchParams.get("state")
        assert.notEqual(redirectUri, null)
        assert.notEqual(state, null)
        await fetch(`${redirectUri}?code=unit-code&state=${state}`)
      })
    },
  })

  const credentials = await flow.login()

  assert.equal(authInfo?.instructions, "test login")
  assert.equal(credentials.access.startsWith("access:unit-code:"), true)
  assert.equal(credentials.refresh, "refresh-token")
  assert.equal(credentials.accountId, "test-account")
})

