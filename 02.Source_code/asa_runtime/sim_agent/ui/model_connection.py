from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .model_auth import model_credential_status


@dataclass(frozen=True, slots=True)
class ModelConnectionStatus:
    provider: str
    model: str
    auth_mode: str
    connected: bool
    connection_label: str
    friendly_message: str
    action_hint: str
    provider_credential_store: Path

    def to_payload(self) -> JsonMap:
        return {
            "provider": self.provider,
            "model": self.model,
            "auth_mode": self.auth_mode,
            "connected": self.connected,
            "connection_label": self.connection_label,
            "friendly_message": self.friendly_message,
            "action_hint": self.action_hint,
            "provider_credential_store": str(self.provider_credential_store),
        }


def model_connection_status(provider: str, model: str, auth_mode: str, api_key_env: str) -> ModelConnectionStatus:
    normalized_provider = provider.strip().lower()
    normalized_mode = auth_mode.strip().lower()
    credential_status = model_credential_status(normalized_provider)
    store = credential_status.provider_credential_store
    match normalized_mode:
        case "none":
            return _connection(
                normalized_provider,
                model,
                normalized_mode,
                True,
                "not required",
                "Model auth is disabled for this provider.",
                "Use /model set if this provider should require OAuth or API key auth.",
                store,
            )
        case "api_key":
            return _api_key_connection(normalized_provider, model, normalized_mode, api_key_env, store)
        case "gateway":
            return _api_key_connection(normalized_provider, model, normalized_mode, api_key_env, store)
        case "oauth":
            return _credential_connection(normalized_provider, model, normalized_mode, store)
        case _:
            return _connection(
                normalized_provider,
                model,
                normalized_mode,
                False,
                "invalid auth mode",
                f"Model auth mode '{normalized_mode}' is not supported.",
                "Run /model set --auth-mode oauth|api_key|gateway|none.",
                store,
            )


def model_connection_status_payload(provider: str, model: str, auth_mode: str, api_key_env: str) -> JsonMap:
    return model_connection_status(provider, model, auth_mode, api_key_env).to_payload()


def _api_key_connection(
    provider: str,
    model: str,
    auth_mode: str,
    api_key_env: str,
    store: Path,
) -> ModelConnectionStatus:
    if os.environ.get(api_key_env):
        return _connection(
            provider,
            model,
            auth_mode,
            True,
            "connected by env",
            f"Model provider {provider} can use ${api_key_env}.",
            "Use /model status or Smoke API to verify the live endpoint.",
            store,
        )
    return _credential_connection(provider, model, auth_mode, store)


def _credential_connection(provider: str, model: str, auth_mode: str, store: Path) -> ModelConnectionStatus:
    credential_status = model_credential_status(provider, store=store)
    if not credential_status.stored:
        return _connection(
            provider,
            model,
            auth_mode,
            False,
            "not connected",
            "Model is not connected. No credential is stored for the selected provider.",
            "Run /login, then use /model set --provider <id> --model <model>.",
            store,
        )
    if not credential_status.logged_in:
        if credential_status.expires is not None and credential_status.expires <= int(time.time() * 1000):
            return _connection(
                provider,
                model,
                auth_mode,
                False,
                "expired",
                "Model is not connected. The stored credential is expired.",
                "Run /login again, then rerun /model status or Smoke API.",
                store,
            )
        return _connection(
            provider,
            model,
            auth_mode,
            False,
            "invalid credential",
            "Model is not connected. The stored credential is missing a usable access token.",
            "Run /login again so ASA can store a valid OAuth or API credential.",
            store,
        )
    return _connection(
        provider,
        model,
        auth_mode,
        True,
        "connected",
        f"Model provider {provider} has an active redacted credential.",
        "Use /model status or Smoke API to verify the live endpoint.",
        store,
    )


def _connection(
    provider: str,
    model: str,
    auth_mode: str,
    connected: bool,
    label: str,
    message: str,
    action_hint: str,
    store: Path,
) -> ModelConnectionStatus:
    return ModelConnectionStatus(
        provider=provider,
        model=model,
        auth_mode=auth_mode,
        connected=connected,
        connection_label=label,
        friendly_message=message,
        action_hint=action_hint,
        provider_credential_store=store,
    )
