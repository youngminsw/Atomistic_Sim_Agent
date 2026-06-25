from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from sim_agent.schemas._parse import JsonMap

from .gateway_client_types import GatewayClientSmokeError

SAFE_ERROR_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_.:/=-]+")


def gateway_url(base_url: str, path: str) -> str:
    if path == "/healthz":
        return f"{base_url.rstrip('/').removesuffix('/v1')}/healthz"
    return f"{base_url.rstrip('/').removesuffix('/v1')}{path}"


def gateway_get_json(url: str, token: str | None, timeout_s: float) -> JsonMap:
    request = urllib.request.Request(url, headers=_headers(token), method="GET")
    return _open_json(request, timeout_s)[1]


def gateway_post_json(
    url: str,
    payload: JsonMap,
    token: str | None,
    timeout_s: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, JsonMap]:
    body = json.dumps(payload).encode("utf-8")
    headers = _headers(token)
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
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
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type or _looks_like_sse(raw):
                return response.status, _json_map_from_sse(raw)
            return response.status, _json_map(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = _json_map(raw) if raw else {}
        raise GatewayClientSmokeError(_http_error_blocker(exc.code, payload)) from exc
    except OSError as exc:
        raise GatewayClientSmokeError("endpoint_unreachable") from exc


def _json_map(raw: str) -> JsonMap:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GatewayClientSmokeError("endpoint_json_object_required") from exc
    if not isinstance(value, dict):
        raise GatewayClientSmokeError("endpoint_json_object_required")
    return value


def _looks_like_sse(raw: str) -> bool:
    stripped = raw.lstrip()
    return stripped.startswith("event:") or stripped.startswith("data:")


def _json_map_from_sse(raw: str) -> JsonMap:
    events = _sse_data_events(raw)
    for event in reversed(events):
        response = event.get("response")
        if isinstance(response, dict) and _non_empty_list(response.get("output")):
            return response
        if _non_empty_list(event.get("output")):
            return event
    calls = _tool_calls_from_sse_events(events)
    if calls:
        return {"output": calls}
    output_text = _output_text_from_sse_events(events)
    if output_text is not None:
        return {"output_text": output_text}
    raise GatewayClientSmokeError("endpoint_sse_response_without_tool_calls")


def _sse_data_events(raw: str) -> list[JsonMap]:
    events: list[JsonMap] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _tool_calls_from_sse_events(events: list[JsonMap]) -> list[JsonMap]:
    calls: list[JsonMap] = []
    arguments_by_item: dict[str, object] = {}
    for event in events:
        item_id = _str_or_empty(event.get("item_id")) or _str_or_empty(event.get("output_item_id"))
        if event.get("type") == "response.function_call_arguments.done" and item_id:
            arguments_by_item[item_id] = event.get("arguments", "")
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}:
            call = dict(item)
            if call.get("status") == "in_progress" and not call.get("arguments"):
                continue
            call_id = _str_or_empty(call.get("id")) or _str_or_empty(call.get("item_id"))
            if call_id and "arguments" not in call and call_id in arguments_by_item:
                call["arguments"] = arguments_by_item[call_id]
            calls.append(call)
    return calls


def _output_text_from_sse_events(events: list[JsonMap]) -> str | None:
    deltas: list[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str):
                return text
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
    if deltas:
        return "".join(deltas)
    return None


def _str_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _non_empty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _http_error_blocker(status: int, payload: JsonMap) -> str:
    error_payload = payload.get("error")
    if not isinstance(error_payload, dict):
        return f"endpoint_http_{status}"
    code = error_payload.get("code")
    if isinstance(code, str) and code:
        return code
    message = error_payload.get("message")
    if isinstance(message, str) and message:
        return f"endpoint_http_{status}:{_safe_error_fragment(message)}"
    return f"endpoint_http_{status}"


def _safe_error_fragment(message: str) -> str:
    return SAFE_ERROR_FRAGMENT_RE.sub("_", message.strip())[:160]
