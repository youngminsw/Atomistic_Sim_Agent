from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolResult
from sim_agent.schemas._parse import JsonMap

from .tool_gateway_policy import DEFAULT_TOOL_GATEWAY_POLICY, ToolGatewayPolicy


@dataclass(frozen=True, slots=True)
class ToolGatewaySessionEvent:
    at: float
    event_type: str
    agent_id: str
    summary: str
    run_id: str
    peer: str | None = None
    artifact_ref: str | None = None


def write_tool_gateway_sessions(
    output_dir: Path,
    run_id: str,
    session_id: str,
    tool_results: tuple[RuntimeToolResult, ...],
    policy: ToolGatewayPolicy = DEFAULT_TOOL_GATEWAY_POLICY,
) -> tuple[str, ...]:
    session_dir = output_dir / "sessions"
    orchestrator_path = session_dir / "orchestrator.jsonl"
    tool_runtime_path = session_dir / "tool_runtime.jsonl"
    _append_session_event(
        orchestrator_path,
            ToolGatewaySessionEvent(
                time.time(),
                "gateway_tool_dispatch_requested",
                "orchestrator",
                f"gateway_request_id={policy.gateway_request_id}; session_id={session_id}; policy={policy.policy_id}",
                run_id,
                peer="tool_runtime",
            ),
    )
    for result in tool_results:
        _append_session_event(
            tool_runtime_path,
            ToolGatewaySessionEvent(
                time.time(),
                "runtime_tool_executed",
                "tool_runtime",
                f"{result.tool_name}:{result.status}",
                run_id,
                peer="orchestrator",
                artifact_ref=result.artifact_ref,
            ),
        )
    _append_session_event(
        orchestrator_path,
        ToolGatewaySessionEvent(
            time.time(),
            "gateway_tool_dispatch_completed",
            "orchestrator",
            ",".join(result.status for result in tool_results),
            run_id,
            peer="tool_runtime",
        ),
    )
    return (str(orchestrator_path), str(tool_runtime_path))


def _append_session_event(path: Path, event: ToolGatewaySessionEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_session_event_payload(event), sort_keys=True) + "\n")


def _session_event_payload(event: ToolGatewaySessionEvent) -> JsonMap:
    payload: dict[str, str | float] = {
        "at": event.at,
        "event_type": event.event_type,
        "agent_id": event.agent_id,
        "summary": event.summary,
        "run_id": event.run_id,
    }
    if event.peer is not None:
        payload["peer"] = event.peer
    if event.artifact_ref is not None:
        payload["artifact_ref"] = event.artifact_ref
    return payload
