from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agents_sdk_runtime.gateway_client import GatewayClientSmokeResult, run_production_gateway_client_smoke
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require


CREDENTIAL_STORE_ENV = "ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE"
SMOKE_OUTPUT_DIR_ENV = "ATOMISTIC_MODEL_GATEWAY_SMOKE_DIR"


@dataclass(frozen=True, slots=True)
class ModelAuthError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class ModelCredentialStatus:
    provider: str
    stored: bool
    logged_in: bool
    expires: int | None
    auth_mode: str
    updated_at_ms: int | None
    credential_store: Path

    def to_payload(self, *, include_credential_store: bool = True) -> JsonMap:
        payload: dict[str, object] = {
            "provider": self.provider,
            "stored": self.stored,
            "logged_in": self.logged_in,
            "expires": self.expires,
            "auth_mode": self.auth_mode,
            "updated_at_ms": self.updated_at_ms,
        }
        if include_credential_store:
            payload["credential_store"] = str(self.credential_store)
        return payload


def login_model_gateway(payload: JsonMap) -> JsonMap:
    provider = as_str(require(payload, "provider"), "provider").strip().lower()
    access_token = as_str(require(payload, "access_token"), "access_token")
    refresh_token = _optional_str(payload, "refresh_token") or access_token
    auth_mode = _optional_str(payload, "auth_mode") or "oauth"
    login_profile = _optional_str(payload, "login_profile")
    expires_in_s = _optional_int(payload, "expires_in_s", 3600)
    if not provider:
        raise ModelAuthError("provider_required")
    if not access_token:
        raise ModelAuthError("access_token_required")
    store = _credential_store_path()
    credentials = {
        "access": access_token,
        "refresh": refresh_token,
        "expires": int(time.time() * 1000) + expires_in_s * 1000,
        "authMode": auth_mode,
    }
    _write_credentials(provider, credentials, store, login_profile=login_profile)
    return {
        "ok": True,
        "provider": provider,
        "logged_in": True,
        "expires": credentials["expires"],
        "credential_store": str(store),
    }


def model_auth_status_payload(*, include_credential_store: bool = True) -> JsonMap:
    store = _credential_store_path()
    items = _read_credentials(store)
    statuses = [
        model_credential_status(provider, store=store, entry=entry)
        for provider, entry in sorted(items.items())
    ]
    providers = [status.to_payload(include_credential_store=False) for status in statuses]
    connected_count = sum(1 for status in statuses if status.logged_in)
    payload: dict[str, object] = {
        "ok": True,
        "providers": providers,
        "connected_provider_count": connected_count,
        "friendly_message": _status_message(connected_count),
        "action_hint": "Run /login, or set a provider token and use /model set before Smoke API.",
    }
    if include_credential_store:
        payload["credential_store"] = str(store)
    return payload


def model_credential_status(
    provider: str,
    *,
    store: Path | None = None,
    entry: JsonMap | None = None,
) -> ModelCredentialStatus:
    resolved_store = store or _credential_store_path()
    resolved_entry = entry if entry is not None else _read_credentials(resolved_store).get(provider)
    if resolved_entry is None:
        return ModelCredentialStatus(
            provider=provider,
            stored=False,
            logged_in=False,
            expires=None,
            auth_mode="oauth",
            updated_at_ms=None,
            credential_store=resolved_store,
        )
    return ModelCredentialStatus(
        provider=provider,
        stored=True,
        logged_in=_entry_is_active(resolved_entry),
        expires=_entry_expires(resolved_entry),
        auth_mode=_entry_auth_mode(resolved_entry),
        updated_at_ms=_int_or_none(resolved_entry.get("updatedAtMs")),
        credential_store=resolved_store,
    )


def run_model_gateway_smoke_from_controller(payload: JsonMap) -> JsonMap:
    endpoint = ModelProviderConfig.from_mapping(as_mapping(require(payload, "llm_endpoint"), "llm_endpoint"))
    request = as_mapping(payload.get("request", payload), "request")
    token = access_token_for_provider(endpoint.provider)
    result = run_production_gateway_client_smoke(
        request,
        endpoint,
        _smoke_output_dir(),
        api_key=token,
        offline=False,
    )
    return _smoke_payload(result)


def _smoke_payload(result: GatewayClientSmokeResult) -> JsonMap:
    return {
        "ok": result.ok,
        "production_smoke": result.production_smoke,
        "provider": result.provider,
        "model": result.model,
        "auth_mode": result.auth_mode,
        "gateway_policy_id": result.gateway_policy_id,
        "gateway_health_ok": result.gateway_health_ok,
        "models_count": result.models_count,
        "endpoint_status": result.endpoint_status,
        "gateway_request_id": result.gateway_request_id,
        "hard_blockers": list(result.blockers),
        "session_files": list(result.session_files),
    }


def access_token_for_provider(provider: str) -> str | None:
    entry = _read_credentials(_credential_store_path()).get(provider)
    if entry is None:
        return None
    credentials = entry.get("credentials")
    if not isinstance(credentials, dict):
        return None
    token = credentials.get("access")
    if isinstance(token, str) and token:
        return token
    return None


def _write_credentials(
    provider: str,
    credentials: JsonMap,
    path: Path,
    *,
    login_profile: str | None = None,
) -> None:
    items = _read_credentials(path)
    entry: dict[str, object] = {
        "provider": provider,
        "credentials": credentials,
        "updatedAtMs": int(time.time() * 1000),
    }
    if login_profile:
        entry["loginProfile"] = login_profile
    items[provider] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _read_credentials(path: Path) -> dict[str, JsonMap]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        raise ModelAuthError("credential_store_object_required")
    items: dict[str, JsonMap] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            items[key] = dict(value)
    return items


def _credential_store_path() -> Path:
    configured = os.environ.get(CREDENTIAL_STORE_ENV)
    if configured:
        return Path(configured)
    return Path.home() / ".atomistic-sim-agent" / "model-gateway-credentials.json"


def _smoke_output_dir() -> Path:
    configured = os.environ.get(SMOKE_OUTPUT_DIR_ENV)
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "evidence" / "controller-model-gateway-smoke"


def _optional_str(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    return as_str(value, field)


def _optional_int(payload: JsonMap, field: str, default: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ModelAuthError(f"{field}_must_be_integer")


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _entry_expires(entry: JsonMap) -> int | None:
    credentials = entry.get("credentials")
    if not isinstance(credentials, dict):
        return None
    return _int_or_none(credentials.get("expires"))


def _entry_auth_mode(entry: JsonMap) -> str:
    credentials = entry.get("credentials")
    if not isinstance(credentials, dict):
        return "oauth"
    value = credentials.get("authMode")
    return value if isinstance(value, str) and value else "oauth"


def _entry_has_access_token(entry: JsonMap) -> bool:
    credentials = entry.get("credentials")
    if not isinstance(credentials, dict):
        return False
    token = credentials.get("access")
    return isinstance(token, str) and len(token) > 0


def _entry_is_active(entry: JsonMap) -> bool:
    if not _entry_has_access_token(entry):
        return False
    expires = _entry_expires(entry)
    return expires is None or expires > int(time.time() * 1000)


def _status_message(connected_count: int) -> str:
    if connected_count > 0:
        return f"Model credentials available for {connected_count} provider(s)."
    return "Model is not connected. Choose a provider and run /login before live model calls."
