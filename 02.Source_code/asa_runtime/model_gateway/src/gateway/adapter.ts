import type { OAuthCredentials } from "../oauth/types.js"
import {
  parseModelProviderConfig,
  toPythonModelProviderAdapter,
  type PythonModelProviderAdapter,
} from "../provider/config.js"

export type GatewayAdapterRequest = {
  readonly model_provider: unknown
  readonly credential?: OAuthCredentials
  readonly api_key?: string
}

export type GatewayAdapterResponse = {
  readonly ok: true
  readonly adapter: PythonModelProviderAdapter
  readonly api_key?: string
  readonly credential_expires?: number
}

export function buildGatewayAdapterResponse(request: GatewayAdapterRequest): GatewayAdapterResponse {
  const adapter = toPythonModelProviderAdapter(parseModelProviderConfig(request.model_provider))
  const response = { ok: true, adapter } as const
  const apiKey = request.api_key ?? request.credential?.access
  const withApiKey = apiKey === undefined ? response : { ...response, api_key: apiKey }
  if (request.credential === undefined) {
    return withApiKey
  }
  return { ...withApiKey, credential_expires: request.credential.expires }
}
