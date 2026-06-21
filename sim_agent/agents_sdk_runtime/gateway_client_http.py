from __future__ import annotations

import json
import urllib.error
import urllib.request

from sim_agent.schemas._parse import JsonMap

from .gateway_client_types import GatewayClientSmokeError


def gateway_url(base_url: str, path: str) -> str:
    if path == "/healthz":
        return f"{base_url.rstrip('/').removesuffix('/v1')}/healthz"
    return f"{base_url.rstrip('/').removesuffix('/v1')}{path}"


def gateway_get_json(url: str, token: str | None, timeout_s: float) -> JsonMap:
    request = urllib.request.Request(url, headers=_headers(token), method="GET")
    return _open_json(request, timeout_s)[1]


def gateway_post_json(url: str, payload: JsonMap, token: str | None, timeout_s: float) -> tuple[int, JsonMap]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=_headers(token), method="POST")
    return _open_json(request, timeout_s)


def _headers(token: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    return headers


def _open_json(request: urllib.request.Request, timeout_s: float) -> tuple[int, JsonMap]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            return response.status, _json_map(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = _json_map(raw) if raw else {}
        error_payload = payload.get("error")
        code = error_payload.get("code") if isinstance(error_payload, dict) else None
        raise GatewayClientSmokeError(str(code or f"endpoint_http_{exc.code}")) from exc
    except OSError as exc:
        raise GatewayClientSmokeError("endpoint_unreachable") from exc


def _json_map(raw: str) -> JsonMap:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise GatewayClientSmokeError("endpoint_json_object_required")
    return value
