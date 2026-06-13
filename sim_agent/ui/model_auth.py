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


def login_model_gateway(payload: JsonMap) -> JsonMap:
    provider = as_str(require(payload, "provider"), "provider").strip().lower()
    access_token = as_str(require(payload, "access_token"), "access_token")
    refresh_token = _optional_str(payload, "refresh_token") or access_token
    auth_mode = _optional_str(payload, "auth_mode") or "oauth"
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
    _write_credentials(provider, credentials, store)
    return {
        "ok": True,
        "provider": provider,
        "logged_in": True,
        "expires": credentials["expires"],
        "credential_store": str(store),
    }


def model_auth_status_payload() -> JsonMap:
    store = _credential_store_path()
    items = _read_credentials(store)
    providers = [
        {
            "provider": provider,
            "logged_in": True,
            "expires": _entry_expires(entry),
            "auth_mode": _entry_auth_mode(entry),
            "updated_at_ms": _int_or_none(entry.get("updatedAtMs")),
        }
        for provider, entry in sorted(items.items())
    ]
    return {"ok": True, "credential_store": str(store), "providers": providers}


def run_model_gateway_smoke_from_controller(payload: JsonMap) -> JsonMap:
    endpoint = ModelProviderConfig.from_mapping(as_mapping(require(payload, "llm_endpoint"), "llm_endpoint"))
    request = as_mapping(payload.get("request", payload), "request")
    token = _access_token(endpoint.provider)
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
        "gateway_health_ok": result.gateway_health_ok,
        "models_count": result.models_count,
        "endpoint_status": result.endpoint_status,
        "gateway_request_id": result.gateway_request_id,
        "hard_blockers": list(result.blockers),
        "session_files": list(result.session_files),
    }


def _access_token(provider: str) -> str | None:
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


def _write_credentials(provider: str, credentials: JsonMap, path: Path) -> None:
    items = _read_credentials(path)
    items[provider] = {
        "provider": provider,
        "credentials": credentials,
        "updatedAtMs": int(time.time() * 1000),
    }
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
