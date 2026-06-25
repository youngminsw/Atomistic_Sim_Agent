from __future__ import annotations

import base64
import json
import os

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.ui.model_auth import access_token_for_provider


def provider_token(endpoint: ModelProviderConfig, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    value = os.environ.get(endpoint.api_key_env)
    if value:
        return value
    return access_token_for_provider(endpoint.provider)


def provider_headers(protocol: str, token: str | None) -> dict[str, str]:
    if protocol != "openai_codex_responses":
        return {}
    headers = {
        "OpenAI-Beta": "responses=experimental",
        "originator": "pi",
    }
    account_id = _codex_account_id(token)
    if account_id:
        headers["chatgpt-account-id"] = account_id
    return headers


def _codex_account_id(token: str | None) -> str:
    if not token:
        return ""
    parts = token.split(".")
    if len(parts) != 3:
        return ""
    try:
        payload_raw = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload_raw.encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    auth = payload.get("https://api.openai.com/auth")
    if not isinstance(auth, dict):
        return ""
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) else ""
