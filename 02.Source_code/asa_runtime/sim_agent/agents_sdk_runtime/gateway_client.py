from __future__ import annotations

import json
import os
from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .gateway_client_policy import (
    DEFAULT_GATEWAY_AGENT_PLAN_POLICY,
    GatewayAgentPlanPolicy,
    gateway_prompt_metadata,
)
from .gateway_client_http import gateway_get_json, gateway_post_json, gateway_url
from .gateway_client_sessions import write_gateway_sessions
from .gateway_client_types import (
    PRODUCTION_GATEWAY_SMOKE_LEDGER_NAME,
    GatewayClientSmokeError,
    GatewayClientSmokeResult,
)


def run_production_gateway_client_smoke(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
    *,
    api_key: str | None = None,
    offline: bool = False,
    timeout_s: float = 10.0,
    policy: GatewayAgentPlanPolicy = DEFAULT_GATEWAY_AGENT_PLAN_POLICY,
) -> GatewayClientSmokeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        return _write_blocked_result(output_dir, endpoint, "production_smoke_refuses_offline", policy)
    token = api_key or _env_token(endpoint)
    if _credentials_required(endpoint) and token is None:
        return _write_blocked_result(output_dir, endpoint, "missing_gateway_credentials", policy)

    task_id = f"gateway-smoke-{_request_id(payload)}"
    try:
        health = gateway_get_json(gateway_url(endpoint.base_url, "/healthz"), token, timeout_s)
        models = gateway_get_json(gateway_url(endpoint.base_url, "/v1/models"), token, timeout_s)
        response_status, response = gateway_post_json(
            gateway_url(endpoint.base_url, "/v1/responses"),
            _gateway_prompt(payload, endpoint, policy),
            token,
            timeout_s,
        )
    except GatewayClientSmokeError as exc:
        return _write_blocked_result(output_dir, endpoint, str(exc), policy)

    gateway_request_id = _optional_str(response, "gateway_request_id")
    try:
        session_files = write_gateway_sessions(output_dir, task_id, gateway_request_id, response, policy)
    except GatewayClientSmokeError as exc:
        return _write_blocked_result(output_dir, endpoint, str(exc), policy)
    result = GatewayClientSmokeResult(
        production_smoke=True,
        offline=False,
        fake_gateway_model=False,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        base_url=endpoint.base_url,
        gateway_policy_id=policy.policy_id,
        gateway_health_ok=health.get("ok") is True,
        models_count=_models_count(models),
        endpoint_status=response_status,
        gateway_request_id=gateway_request_id,
        blockers=_gateway_success_blockers(health, models, response_status, gateway_request_id),
        session_files=tuple(str(path) for path in session_files),
        final_output=_response_text(response),
    )
    write_production_gateway_smoke_ledger(output_dir, result)
    return result


def write_production_gateway_smoke_ledger(output_dir: Path, result: GatewayClientSmokeResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PRODUCTION_GATEWAY_SMOKE_LEDGER_NAME
    payload = production_gateway_smoke_payload(result)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "gateway_policy_id": result.gateway_policy_id,
        "gateway_health_ok": result.gateway_health_ok,
        "models_count": result.models_count,
        "endpoint_status": result.endpoint_status,
        "gateway_request_id": result.gateway_request_id,
        "hard_blockers": list(result.blockers),
        "session_files": list(result.session_files),
        "final_output": result.final_output,
    }


def _write_blocked_result(
    output_dir: Path,
    endpoint: ModelProviderConfig,
    blocker: str,
    policy: GatewayAgentPlanPolicy,
) -> GatewayClientSmokeResult:
    result = _blocked_result(endpoint, blocker, policy)
    write_production_gateway_smoke_ledger(output_dir, result)
    return result


def _blocked_result(
    endpoint: ModelProviderConfig,
    blocker: str,
    policy: GatewayAgentPlanPolicy,
) -> GatewayClientSmokeResult:
    return GatewayClientSmokeResult(
        production_smoke=True,
        offline=blocker == "production_smoke_refuses_offline",
        fake_gateway_model=False,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        base_url=endpoint.base_url,
        gateway_policy_id=policy.policy_id,
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


def _gateway_prompt(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    policy: GatewayAgentPlanPolicy,
) -> JsonMap:
    request_id = _request_id(payload)
    return {
        "model": endpoint.model,
        "input": [
            {
                "role": "user",
                "content": _user_goal(payload),
            }
        ],
        "metadata": gateway_prompt_metadata(policy, request_id),
    }


def _gateway_success_blockers(
    health: JsonMap,
    models: JsonMap,
    response_status: int,
    gateway_request_id: str | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if health.get("ok") is not True:
        blockers.append("gateway_health_check_failed")
    if _models_count(models) <= 0:
        blockers.append("gateway_models_empty")
    if response_status != 200 or gateway_request_id is None:
        blockers.append("endpoint_response_missing_gateway_request_id")
    return tuple(blockers)


def _models_count(payload: JsonMap) -> int:
    data = payload.get("data")
    return len(data) if isinstance(data, list) else 0


def _optional_str(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    return value if isinstance(value, str) and value else None


def _response_text(payload: JsonMap) -> str:
    value = payload.get("output_text") or payload.get("final_output")
    return value if isinstance(value, str) else "gateway_response_received"


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    return value if isinstance(value, str) and value else "anonymous"


def _user_goal(payload: JsonMap) -> str:
    value = payload.get("user_goal") or payload.get("goal")
    return value if isinstance(value, str) and value else f"Run simulation request {_request_id(payload)}"
