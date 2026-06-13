import type { OAuthCredentials, OAuthLoginCallbacks, OAuthProviderId, OAuthProviderInterface } from "../oauth/types.js"

export type FakeOAuthProviderOptions = {
  readonly id: string
  readonly name?: string
  readonly authUrl?: string
  readonly accessToken?: string
  readonly refreshToken?: string
  readonly expiresInMs?: number
  readonly accountId?: string
  readonly email?: string
}

export class FakeOAuthProvider implements OAuthProviderInterface {
  readonly id: OAuthProviderId
  readonly name: string
  readonly sourceId = "atomistic-sim-agent-test"
  readonly #options: FakeOAuthProviderOptions

  constructor(options: FakeOAuthProviderOptions) {
    this.id = options.id as OAuthProviderId
    this.name = options.name ?? options.id
    this.#options = options
  }

  async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
    callbacks.onAuth({ url: this.#options.authUrl ?? "https://auth.example.test/login" })
    callbacks.onProgress?.(`Logged in to ${this.name}`)
    return this.#credentials(this.#options.accessToken ?? `access:${this.id}`)
  }

  async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
    const refreshedAccess = `${credentials.access}:refreshed`
    return this.#credentials(refreshedAccess, credentials.refresh)
  }

  getApiKey(credentials: OAuthCredentials): string {
    return credentials.access
  }

  #credentials(access: string, refresh?: string): OAuthCredentials {
    const base = {
      access,
      refresh: refresh ?? this.#options.refreshToken ?? `refresh:${this.id}`,
      expires: Date.now() + (this.#options.expiresInMs ?? 60_000),
    }
    const withAccount = this.#options.accountId === undefined ? base : { ...base, accountId: this.#options.accountId }
    if (this.#options.email === undefined) {
      return withAccount
    }
    return { ...withAccount, email: this.#options.email }
  }
}
