from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .gateway_client_policy import (
    DEFAULT_GATEWAY_AGENT_PLAN_POLICY,
    GatewayAgentPlanPolicy,
    gateway_agent_allowed,
    gateway_agent_plan,
)
from .gateway_client_types import GatewayClientSmokeError, GatewaySessionEvent


def write_gateway_sessions(
    output_dir: Path,
    task_id: str,
    gateway_request_id: str | None,
    response: JsonMap,
    policy: GatewayAgentPlanPolicy = DEFAULT_GATEWAY_AGENT_PLAN_POLICY,
) -> tuple[Path, ...]:
    session_dir = output_dir / "sessions"
    specialist, second_call = gateway_agent_plan(policy, response)
    files = [
        _append_session(
            session_dir,
            "orchestrator",
            policy,
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
            policy,
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
            policy,
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


def _append_session(
    session_dir: Path,
    agent_id: str,
    policy: GatewayAgentPlanPolicy,
    event: GatewaySessionEvent,
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_session_file(session_dir, agent_id, policy)
    payload = {key: value for key, value in asdict(event).items() if value is not None}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _safe_session_file(session_dir: Path, agent_id: str, policy: GatewayAgentPlanPolicy) -> Path:
    if not gateway_agent_allowed(policy, agent_id):
        raise GatewayClientSmokeError("gateway_agent_plan_invalid")
    root = session_dir.resolve()
    path = (root / f"{agent_id}.jsonl").resolve()
    if root not in path.parents:
        raise GatewayClientSmokeError("gateway_session_path_invalid")
    return path
