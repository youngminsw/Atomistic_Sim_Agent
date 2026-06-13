export class OAuthCallbackError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = "OAuthCallbackError"
    this.code = code
  }
}

export class OAuthTokenExchangeError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = "OAuthTokenExchangeError"
    if (status !== undefined) {
      this.status = status
    }
  }
}

