import { createServer, type Server } from "node:http"
import type { AddressInfo } from "node:net"
import { webcrypto } from "node:crypto"
import { OAuthCallbackError } from "./errors.js"
import { renderOAuthResultHtml } from "./html.js"
import type { CallbackRenderState, CallbackResult, OAuthController, OAuthCredentials } from "./types.js"

const DEFAULT_TIMEOUT_MS = 300_000
const DEFAULT_HOSTNAME = "localhost"
const CALLBACK_PATH = "/callback"

export type OAuthCallbackFlowOptions = {
  readonly preferredPort: number
  readonly callbackPath?: string
  readonly callbackHostname?: string
  readonly callbackBindHostname?: string
  readonly redirectUri?: string
}

export abstract class OAuthCallbackFlow {
  readonly ctrl: OAuthController
  readonly preferredPort: number
  readonly callbackPath: string
  readonly callbackHostname: string
  readonly callbackBindHostname: string
  readonly redirectUri: string | undefined
  #callbackResolve?: (result: CallbackResult) => void
  #callbackReject?: (error: Error) => void

  constructor(ctrl: OAuthController, options: number | OAuthCallbackFlowOptions) {
    this.ctrl = ctrl
    if (typeof options === "number") {
      this.preferredPort = options
      this.callbackPath = CALLBACK_PATH
      this.callbackHostname = DEFAULT_HOSTNAME
      this.callbackBindHostname = DEFAULT_HOSTNAME
      return
    }
    this.preferredPort = options.preferredPort
    this.callbackPath = options.callbackPath ?? CALLBACK_PATH
    this.callbackHostname = options.callbackHostname ?? DEFAULT_HOSTNAME
    this.callbackBindHostname = options.callbackBindHostname ?? this.callbackHostname
    this.redirectUri = options.redirectUri
  }

  abstract generateAuthUrl(
    state: string,
    redirectUri: string,
  ): Promise<{ readonly url: string; readonly instructions?: string }>

  abstract exchangeToken(
    code: string,
    state: string,
    redirectUri: string,
  ): Promise<OAuthCredentials>

  generateState(): string {
    const bytes = new Uint8Array(16)
    webcrypto.getRandomValues(bytes)
    return Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("")
  }

  async login(): Promise<OAuthCredentials> {
    const state = this.generateState()
    const { server, redirectUri } = await this.#startCallbackServer(state)
    try {
      const { url, instructions } = await this.generateAuthUrl(state, redirectUri)
      this.ctrl.onAuth?.(instructions === undefined ? { url } : { url, instructions })
      this.ctrl.onProgress?.("Waiting for browser authentication...")
      const callback = await this.#waitForCallback(state)
      this.ctrl.onProgress?.("Exchanging authorization code for tokens...")
      return await this.exchangeToken(callback.code, callback.state, redirectUri)
    } finally {
      await closeServer(server)
    }
  }

  async #startCallbackServer(expectedState: string): Promise<{ readonly server: Server; readonly redirectUri: string }> {
    try {
      const server = await listenServer(this.preferredPort, this.callbackBindHostname, req =>
        this.#handleCallback(req.url, expectedState),
      )
      if (this.redirectUri !== undefined) {
        return { server, redirectUri: this.redirectUri }
      }
      const port = getListeningPort(server)
      return {
        server,
        redirectUri: `http://${this.callbackHostname}:${port}${this.callbackPath}`,
      }
    } catch (error) {
      if (this.redirectUri !== undefined) {
        throw normalizeCallbackError(error, "callback_port_unavailable")
      }
      const server = await listenServer(0, this.callbackBindHostname, req => this.#handleCallback(req.url, expectedState))
      const port = getListeningPort(server)
      this.ctrl.onProgress?.(`Preferred port ${this.preferredPort} unavailable, using port ${port}`)
      return { server, redirectUri: `http://${this.callbackHostname}:${port}${this.callbackPath}` }
    }
  }

  #handleCallback(rawUrl: string | undefined, expectedState: string): CallbackRenderState {
    if (rawUrl === undefined) {
      return this.#rejectCallback("Missing callback URL")
    }
    const url = new URL(rawUrl, `http://${this.callbackHostname}`)
    if (url.pathname !== this.callbackPath) {
      return this.#rejectCallback("Unexpected callback path")
    }
    const error = url.searchParams.get("error")
    const code = url.searchParams.get("code")
    const state = url.searchParams.get("state") ?? ""
    if (error !== null) {
      return this.#rejectCallback(url.searchParams.get("error_description") ?? error)
    }
    if (code === null || code.length === 0) {
      return this.#rejectCallback("Missing authorization code")
    }
    if (expectedState.length > 0 && state !== expectedState) {
      return this.#rejectCallback("State mismatch - possible CSRF attack")
    }
    queueMicrotask(() => this.#callbackResolve?.({ code, state }))
    return { ok: true, code, state }
  }

  #rejectCallback(message: string): CallbackRenderState {
    const error = new OAuthCallbackError("callback_rejected", message)
    queueMicrotask(() => this.#callbackReject?.(error))
    return { ok: false, error: message }
  }

  #waitForCallback(expectedState: string): Promise<CallbackResult> {
    const callbackPromise = new Promise<CallbackResult>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new OAuthCallbackError("callback_timeout", "OAuth callback timed out"))
      }, DEFAULT_TIMEOUT_MS)
      const abort = (): void => reject(new OAuthCallbackError("callback_cancelled", "OAuth callback cancelled"))
      this.ctrl.signal?.addEventListener("abort", abort, { once: true })
      this.#callbackResolve = result => {
        clearTimeout(timeout)
        this.ctrl.signal?.removeEventListener("abort", abort)
        resolve(result)
      }
      this.#callbackReject = error => {
        clearTimeout(timeout)
        this.ctrl.signal?.removeEventListener("abort", abort)
        reject(error)
      }
    })
    if (this.ctrl.onManualCodeInput === undefined) {
      return callbackPromise
    }
    return Promise.race([callbackPromise, waitForManualCode(this.ctrl.onManualCodeInput, expectedState)])
  }
}

async function waitForManualCode(requestManualInput: () => Promise<string>, expectedState: string): Promise<CallbackResult> {
  for (;;) {
    const parsed = parseCallbackInput(await requestManualInput())
    if (parsed.code !== undefined && (expectedState.length === 0 || parsed.state === undefined || parsed.state === expectedState)) {
      return { code: parsed.code, state: parsed.state ?? "" }
    }
  }
}

export function parseCallbackInput(input: string): { readonly code?: string; readonly state?: string } {
  const value = input.trim()
  if (value.length === 0) {
    return {}
  }
  try {
    const url = new URL(value)
    return optionalCallbackParts(url.searchParams.get("code"), url.searchParams.get("state"))
  } catch (error) {
    if (!(error instanceof TypeError)) {
      throw error
    }
  }
  if (value.includes("code=")) {
    const params = new URLSearchParams(value.replace(/^[?#]/, ""))
    return optionalCallbackParts(params.get("code"), params.get("state"))
  }
  const [code, state] = value.split("#", 2)
  return optionalCallbackParts(code, state ?? null)
}

function optionalCallbackParts(code: string | null | undefined, state: string | null | undefined): { readonly code?: string; readonly state?: string } {
  const result: { code?: string; state?: string } = {}
  if (code !== null && code !== undefined && code.length > 0) {
    result.code = code
  }
  if (state !== null && state !== undefined && state.length > 0) {
    result.state = state
  }
  return result
}

async function listenServer(
  port: number,
  hostname: string,
  handle: (request: { readonly url?: string }) => CallbackRenderState,
): Promise<Server> {
  const server = createServer((request, response) => {
    const state = handle(request.url === undefined ? {} : { url: request.url })
    response.writeHead(state.ok ? 200 : 500, { "content-type": "text/html; charset=utf-8" })
    response.end(renderOAuthResultHtml(state))
  })
  return await new Promise<Server>((resolve, reject) => {
    server.once("error", reject)
    server.listen(port, hostname, () => {
      server.off("error", reject)
      resolve(server)
    })
  })
}

function getListeningPort(server: Server): number {
  const address = server.address()
  if (isAddressInfo(address)) {
    return address.port
  }
  throw new OAuthCallbackError("callback_port_unknown", "Callback server did not expose a TCP port")
}

function isAddressInfo(value: AddressInfo | string | null): value is AddressInfo {
  return typeof value === "object" && value !== null && typeof value.port === "number"
}

async function closeServer(server: Server): Promise<void> {
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

function normalizeCallbackError(error: unknown, code: string): Error {
  if (error instanceof Error) {
    return error
  }
  return new OAuthCallbackError(code, String(error))
}
