from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap, as_mapping

from .graph_memory import runtime_graph_memory_payload
from .roles import AGENT_ROLES
from .sdk_bridge import agents_sdk_available, run_agents_sdk_fake_gateway_smoke
from .session_files import ensure_runtime_session_path
from .skill_registry import run_registered_agent_skills, skill_registry_summary
from .types import (
    AgentsSdkRuntimeResult,
    ApprovalGate,
    ApprovalStatus,
    RuntimeMessage,
    RuntimeTraceEvent,
)


AGENTS_SDK_RUNTIME_LEDGER_NAME = "agents_sdk_runtime_ledger.json"


def run_agents_sdk_runtime_dry_run(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    *,
    agent_model_assignments: JsonMap | None = None,
    run_sdk_smoke: bool = False,
    output_dir: Path | None = None,
) -> AgentsSdkRuntimeResult:
    request_id = _request_id(payload)
    session_id = f"sim-agent-sdk-{request_id}"
    session_path = ensure_runtime_session_path(session_id, output_dir)
    sdk_available = agents_sdk_available()
    sdk_output = ""
    if run_sdk_smoke:
        sdk_output = run_agents_sdk_fake_gateway_smoke(endpoint, session_id, _user_goal(payload), session_path)
    skill_invocations = run_registered_agent_skills(payload, output_dir=output_dir)
    messages = _message_log()
    approvals = _approval_gates(payload)
    graph_memory = runtime_graph_memory_payload(payload, ("orchestrator",) + tuple(role.role_id for role in AGENT_ROLES))
    model_assignments = agent_model_assignments or _default_agent_model_assignments(endpoint)
    trace = _trace_events(session_id, sdk_available, bool(sdk_output), approvals)
    return AgentsSdkRuntimeResult(
        run_id=f"agents-sdk-{request_id}",
        session_id=session_id,
        sdk_available=sdk_available,
        sdk_run_completed=bool(sdk_output),
        provider=endpoint.provider,
        model=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        auth_mode=endpoint.auth_mode,
        session_path=str(session_path),
        handoff_sequence=tuple(role.role_id for role in AGENT_ROLES),
        agent_model_assignments=model_assignments,
        messages=messages,
        trace=trace,
        approval_gates=approvals,
        graph_memory=graph_memory,
        skill_registry=skill_registry_summary(),
        skill_invocations=skill_invocations,
        final_output=sdk_output or "agents_sdk_runtime_dry_run_planned",
    )


def write_agents_sdk_runtime_ledger(output_dir: Path, result: AgentsSdkRuntimeResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / AGENTS_SDK_RUNTIME_LEDGER_NAME
    ledger_path.write_text(json.dumps(agents_sdk_runtime_payload(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger_path


def agents_sdk_runtime_payload(result: AgentsSdkRuntimeResult) -> JsonMap:
    return {
        "ledger_version": "agents_sdk_runtime_v1",
        "run_id": result.run_id,
        "session_id": result.session_id,
        "sdk_available": result.sdk_available,
        "sdk_run_completed": result.sdk_run_completed,
        "provider": result.provider,
        "model": result.model,
        "reasoning_effort": result.reasoning_effort,
        "auth_mode": result.auth_mode,
        "session_path": result.session_path,
        "handoff_sequence": list(result.handoff_sequence),
        "agent_model_assignments": result.agent_model_assignments,
        "messages": [asdict(message) for message in result.messages],
        "trace": [asdict(event) for event in result.trace],
        "approval_gates": [
            {
                "gate_id": gate.gate_id,
                "status": gate.status.value,
                "reason": gate.reason,
            }
            for gate in result.approval_gates
        ],
        "graph_memory": result.graph_memory,
        "skill_registry": result.skill_registry,
        "skill_invocations": [asdict(invocation) for invocation in result.skill_invocations],
        "final_output": result.final_output,
    }


def _message_log() -> tuple[RuntimeMessage, ...]:
    messages: list[RuntimeMessage] = []
    for role in AGENT_ROLES:
        messages.append(RuntimeMessage("orchestrator", role.role_id, f"handoff:{role.boundary}"))
        messages.append(RuntimeMessage(role.role_id, "orchestrator", f"ack:{role.role_id}"))
    return tuple(messages)


def _default_agent_model_assignments(endpoint: ModelProviderConfig) -> JsonMap:
    return {
        role.role_id: {
            "provider": endpoint.provider,
            "model": endpoint.model,
            "reasoning_effort": endpoint.reasoning_effort,
            "base_url": endpoint.base_url,
            "auth_mode": endpoint.auth_mode,
            "api_key_env": endpoint.api_key_env,
            "source": "default",
        }
        for role in AGENT_ROLES
    }


def _trace_events(
    session_id: str,
    sdk_available: bool,
    sdk_run_completed: bool,
    approvals: tuple[ApprovalGate, ...],
) -> tuple[RuntimeTraceEvent, ...]:
    events = [
        RuntimeTraceEvent("session_created", "orchestrator", session_id),
        RuntimeTraceEvent("sdk_available", "orchestrator", str(sdk_available).lower()),
        RuntimeTraceEvent("sdk_run_completed", "orchestrator", str(sdk_run_completed).lower()),
        RuntimeTraceEvent("skill_registry_loaded", "orchestrator", "callable_handlers"),
    ]
    events.extend(
        RuntimeTraceEvent("handoff_registered", "orchestrator", f"{role.handoff_tool_name}:{role.role_id}")
        for role in AGENT_ROLES
    )
    events.extend(RuntimeTraceEvent("approval_gate", "orchestrator", f"{gate.gate_id}:{gate.status.value}") for gate in approvals)
    events.append(RuntimeTraceEvent("qa_review_required", "qa_agent", "hard_blockers_checked_before_acceptance"))
    return tuple(events)


def _approval_gates(payload: JsonMap) -> tuple[ApprovalGate, ...]:
    approvals = as_mapping(payload.get("approvals", {}), "approvals")
    return tuple(
        gate
        for gate in (
            _approval_gate(
                "remote_execution",
                _remote_requested(payload),
                bool(approvals.get("remote_execution")),
                "remote worker/server use requires user approval",
            ),
            _approval_gate(
                "long_runtime",
                _estimated_runtime_seconds(payload) > 3600,
                bool(approvals.get("long_runtime")),
                "estimated runtime over one hour requires user approval",
            ),
            _approval_gate(
                "graphdb_write",
                _graphdb_write_requested(payload),
                bool(approvals.get("graphdb_write")),
                "Neo4j writes require explicit user approval",
            ),
            _approval_gate(
                "destructive_action",
                payload.get("destructive_action") is True,
                bool(approvals.get("destructive_action")),
                "destructive actions require explicit user approval",
            ),
        )
    )


def _approval_gate(gate_id: str, required: bool, approved: bool, reason: str) -> ApprovalGate:
    if approved:
        return ApprovalGate(gate_id, ApprovalStatus.APPROVED, reason)
    if required:
        return ApprovalGate(gate_id, ApprovalStatus.REQUIRED, reason)
    return ApprovalGate(gate_id, ApprovalStatus.NOT_REQUIRED, reason)


def _remote_requested(payload: JsonMap) -> bool:
    if payload.get("remote_execution") is True:
        return True
    host = payload.get("host")
    return isinstance(host, str) and host not in {"", "local", "localhost"}


def _estimated_runtime_seconds(payload: JsonMap) -> float:
    value = payload.get("estimated_runtime_s", payload.get("estimated_runtime_seconds", 0.0))
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _graphdb_write_requested(payload: JsonMap) -> bool:
    graphdb = payload.get("graphdb")
    if not isinstance(graphdb, dict):
        return False
    return graphdb.get("mode") in {"attempt_write", "write"}


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def _user_goal(payload: JsonMap) -> str:
    value = payload.get("user_goal")
    if isinstance(value, str) and value:
        return value
    return f"Run atomistic simulation request {_request_id(payload)}"
