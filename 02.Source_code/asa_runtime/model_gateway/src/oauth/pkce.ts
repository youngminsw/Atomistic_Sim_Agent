import { Buffer } from "node:buffer"
import { webcrypto } from "node:crypto"

export type PKCEPair = {
  readonly verifier: string
  readonly challenge: string
}

export async function generatePKCE(): Promise<PKCEPair> {
  const verifierBytes = new Uint8Array(96)
  webcrypto.getRandomValues(verifierBytes)
  const verifier = Buffer.from(verifierBytes).toString("base64url")
  const encoded = new TextEncoder().encode(verifier)
  const hashBuffer = await webcrypto.subtle.digest("SHA-256", encoded)
  const challenge = Buffer.from(hashBuffer).toString("base64url")
  return { verifier, challenge }
}
