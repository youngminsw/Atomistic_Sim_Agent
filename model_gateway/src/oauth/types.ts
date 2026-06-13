export type OAuthCredentials = {
  readonly refresh: string
  readonly access: string
  readonly expires: number
  readonly enterpriseUrl?: string
  readonly projectId?: string
  readonly email?: string
  readonly accountId?: string
}

export type OAuthProviderId = string & { readonly __brand?: "OAuthProviderId" }

export type OAuthPrompt = {
  readonly message: string
  readonly placeholder?: string
  readonly allowEmpty?: boolean
}

export type OAuthAuthInfo = {
  readonly url: string
  readonly instructions?: string
}

export type OAuthProviderInfo = {
  readonly id: OAuthProviderId
  readonly name: string
  readonly available: boolean
}

export interface OAuthController {
  readonly onAuth?: (info: OAuthAuthInfo) => void
  readonly onProgress?: (message: string) => void
  readonly onManualCodeInput?: () => Promise<string>
  readonly onPrompt?: (prompt: OAuthPrompt) => Promise<string>
  readonly signal?: AbortSignal
}

export interface OAuthLoginCallbacks extends OAuthController {
  readonly onAuth: (info: OAuthAuthInfo) => void
  readonly onPrompt: (prompt: OAuthPrompt) => Promise<string>
}

export interface OAuthProviderInterface {
  readonly id: OAuthProviderId
  readonly name: string
  readonly sourceId?: string
  login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials | string>
  refreshToken?(credentials: OAuthCredentials): Promise<OAuthCredentials>
  getApiKey?(credentials: OAuthCredentials): string
}

export type CallbackResult = {
  readonly code: string
  readonly state: string
}

export type CallbackRenderState =
  | {
      readonly ok: true
      readonly code: string
      readonly state: string
    }
  | {
      readonly ok: false
      readonly error: string
    }

