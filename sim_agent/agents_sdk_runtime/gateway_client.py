from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap


PRODUCTION_GATEWAY_SMOKE_LEDGER_NAME = "production_gateway_smoke_ledger.json"


@dataclass(frozen=True, slots=True)
class GatewaySessionEvent:
    at: float
    event_type: str
    agent_id: str
    summary: str
    task_id: str
    peer: str | None = None
    artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class GatewayClientSmokeResult:
    production_smoke: bool
    offline: bool
    fake_gateway_model: bool
    provider: str
    model: str
    auth_mode: str
    base_url: str
    gateway_health_ok: bool
    models_count: int
    endpoint_status: int | None
    gateway_request_id: str | None
    blockers: tuple[str, ...]
    session_files: tuple[str, ...]
    final_output: str

    @property
    def ok(self) -> bool:
        return not self.blockers and self.endpoint_status == 200 and bool(self.gateway_request_id)


class GatewayClientSmokeError(RuntimeError):
    pass


def run_production_gateway_client_smoke(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
    *,
    api_key: str | None = None,
    offline: bool = False,
    timeout_s: float = 10.0,
) -> GatewayClientSmokeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        result = _blocked_result(endpoint, "production_smoke_refuses_offline")
        write_production_gateway_smoke_ledger(output_dir, result)
        return result
    token = api_key or _env_token(endpoint)
    if _credentials_required(endpoint) and token is None:
        result = _blocked_result(endpoint, "missing_gateway_credentials")
        write_production_gateway_smoke_ledger(output_dir, result)
        return result

    task_id = f"gateway-smoke-{_request_id(payload)}"
    try:
        health = _get_json(_url(endpoint.base_url, "/healthz"), token, timeout_s)
        models = _get_json(_url(endpoint.base_url, "/v1/models"), token, timeout_s)
        response_status, response = _post_json(
            _url(endpoint.base_url, "/v1/responses"),
            _gateway_prompt(payload, endpoint),
            token,
            timeout_s,
        )
    except GatewayClientSmokeError as exc:
        result = _blocked_result(endpoint, str(exc))
        write_production_gateway_smoke_ledger(output_dir, result)
        return result

    gateway_request_id = _optional_str(response, "gateway_request_id")
    session_files = _write_gateway_sessions(output_dir, task_id, gateway_request_id, response)
    output = _response_text(response)
    result = GatewayClientSmokeResult(
        production_smoke=True,
        offline=False,
        fake_gateway_model=False,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        base_url=endpoint.base_url,
        gateway_health_ok=health.get("ok") is True,
        models_count=_models_count(models),
        endpoint_status=response_status,
        gateway_request_id=gateway_request_id,
        blockers=() if response_status == 200 and gateway_request_id else ("endpoint_response_missing_gateway_request_id",),
        session_files=tuple(str(path) for path in session_files),
        final_output=output,
    )
    write_production_gateway_smoke_ledger(output_dir, result)
    return result


def write_production_gateway_smoke_ledger(output_dir: Path, result: GatewayClientSmokeResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PRODUCTION_GATEWAY_SMOKE_LEDGER_NAME
    path.write_text(json.dumps(production_gateway_smoke_payload(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def production_gateway_smoke_payload(result: GatewayClientSmokeResult) -> dict[str, object]:
    return {
        "ledger_version": "production_gateway_smoke_v1",
        "production_smoke": result.production_smoke,
        "offline": result.offline,
        "fake_gateway_model": result.fake_gateway_model,
        "provider": result.provider,
        "model": result.model,
        "auth_mode": result.auth_mode,
        "base_url": result.base_url,
        "gateway_health_ok": result.gateway_health_ok,
        "models_count": result.models_count,
        "endpoint_status": result.endpoint_status,
        "gateway_request_id": result.gateway_request_id,
        "hard_blockers": list(result.blockers),
        "session_files": list(result.session_files),
        "final_output": result.final_output,
    }


def _blocked_result(endpoint: ModelProviderConfig, blocker: str) -> GatewayClientSmokeResult:
    return GatewayClientSmokeResult(
        production_smoke=True,
        offline=blocker == "production_smoke_refuses_offline",
        fake_gateway_model=False,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        base_url=endpoint.base_url,
        gateway_health_ok=False,
        models_count=0,
        endpoint_status=None,
        gateway_request_id=None,
        blockers=(blocker,),
        session_files=(),
        final_output="blocked",
    )


def _env_token(endpoint: ModelProviderConfig) -> str | None:
    value = os.environ.get(endpoint.api_key_env)
    if value:
        return value
    return None


def _credentials_required(endpoint: ModelProviderConfig) -> bool:
    return endpoint.auth_mode in {"api_key", "oauth", "gateway"}


def _url(base_url: str, path: str) -> str:
    if path == "/healthz":
        return f"{base_url.rstrip('/').removesuffix('/v1')}/healthz"
    return f"{base_url.rstrip('/').removesuffix('/v1')}{path}"


def _get_json(url: str, token: str | None, timeout_s: float) -> JsonMap:
    request = urllib.request.Request(url, headers=_headers(token), method="GET")
    return _open_json(request, timeout_s)[1]


def _post_json(url: str, payload: JsonMap, token: str | None, timeout_s: float) -> tuple[int, JsonMap]:
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
        code = payload.get("error", {}).get("code") if isinstance(payload.get("error"), dict) else None
        raise GatewayClientSmokeError(str(code or f"endpoint_http_{exc.code}")) from exc
    except OSError as exc:
        raise GatewayClientSmokeError("endpoint_unreachable") from exc


def _json_map(raw: str) -> JsonMap:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise GatewayClientSmokeError("endpoint_json_object_required")
    return value


def _gateway_prompt(payload: JsonMap, endpoint: ModelProviderConfig) -> JsonMap:
    return {
        "model": endpoint.model,
        "input": [
            {
                "role": "user",
                "content": _user_goal(payload),
            }
        ],
        "metadata": {
            "simulation_request_id": _request_id(payload),
            "required_call_graph": [
                "orchestrator->research_graphdb_agent",
                "research_graphdb_agent->qa_agent",
            ],
            "response_contract": {
                "agent_plan.specialist": "research_graphdb_agent",
                "agent_plan.second_call": "qa_agent",
            },
        },
    }


def _write_gateway_sessions(
    output_dir: Path,
    task_id: str,
    gateway_request_id: str | None,
    response: JsonMap,
) -> tuple[Path, ...]:
    session_dir = output_dir / "sessions"
    specialist, second_call = _agent_plan(response)
    files = [
        _append_session(
            session_dir,
            "orchestrator",
            GatewaySessionEvent(
                time.time(),
                "gateway_endpoint_call",
                "orchestrator",
                f"gateway_request_id={gateway_request_id}",
                task_id,
                peer=specialist,
            ),
        ),
        _append_session(
            session_dir,
            specialist,
            GatewaySessionEvent(
                time.time(),
                "agent_call_received",
                specialist,
                "endpoint response selected specialist",
                task_id,
                peer="orchestrator",
            ),
        ),
        _append_session(
            session_dir,
            second_call,
            GatewaySessionEvent(
                time.time(),
                "qa_or_research_check_requested",
                second_call,
                "specialist requested downstream validation",
                task_id,
                peer=specialist,
            ),
        ),
    ]
    return tuple(files)


def _append_session(session_dir: Path, agent_id: str, event: GatewaySessionEvent) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{agent_id}.jsonl"
    payload = {key: value for key, value in asdict(event).items() if value is not None}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _agent_plan(response: JsonMap) -> tuple[str, str]:
    plan = response.get("agent_plan")
    if isinstance(plan, dict):
        specialist = plan.get("specialist")
        second_call = plan.get("second_call")
        if isinstance(specialist, str) and isinstance(second_call, str):
            return specialist, second_call
    return "research_graphdb_agent", "qa_agent"


def _models_count(payload: JsonMap) -> int:
    data = payload.get("data")
    return len(data) if isinstance(data, list) else 0


def _optional_str(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    return value if isinstance(value, str) and value else None


def _response_text(payload: JsonMap) -> str:
    value = payload.get("output_text") or payload.get("final_output")
    return value if isinstance(value, str) and value else "gateway_response_received"


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    return value if isinstance(value, str) and value else "anonymous"


def _user_goal(payload: JsonMap) -> str:
    value = payload.get("user_goal") or payload.get("goal")
    return value if isinstance(value, str) and value else f"Run simulation request {_request_id(payload)}"
