from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap, as_mapping

from .session_files import ensure_runtime_session_path
from .skill_registry import run_registered_agent_skills, skill_registry_summary
from .types import (
    AgentRoleDefinition,
    AgentsSdkRuntimeResult,
    AgentsSdkTeam,
    ApprovalGate,
    ApprovalStatus,
    RuntimeMessage,
    RuntimeTraceEvent,
)


AGENTS_SDK_RUNTIME_LEDGER_NAME = "agents_sdk_runtime_ledger.json"


class AgentsSdkRuntimeError(RuntimeError):
    pass


AGENT_ROLES: tuple[AgentRoleDefinition, ...] = (
    AgentRoleDefinition(
        "md_agent",
        "MD Agent",
        "LAMMPS MD structure/build/run/postprocess/physics gates",
        "handoff_to_md_agent",
        "Plan and verify MD work. Never bypass force-field, box-size, physics, or event-quality gates.",
    ),
    AgentRoleDefinition(
        "ml_mdn_agent",
        "ML/MDN Agent",
        "MD event dataset audit, MDN training, uncertainty, active learning",
        "handoff_to_ml_mdn_agent",
        "Train and gate MD-derived surrogate models before feature-scale use.",
    ),
    AgentRoleDefinition(
        "feature_scale_agent",
        "Feature Scale Agent",
        "KMC transport and Level-Set profile evolution",
        "handoff_to_feature_scale_agent",
        "Convert MDN outputs and plasma distributions into profile evolution artifacts.",
    ),
    AgentRoleDefinition(
        "research_graphdb_agent",
        "Research GraphDB Agent",
        "Literature search, source-to-graph ingestion, provenance retrieval",
        "handoff_to_research_graphdb_agent",
        "Build source-backed knowledge with explicit Neo4j write approval boundaries.",
    ),
    AgentRoleDefinition(
        "qa_agent",
        "QA Agent",
        "Run evidence audit, hard blocker checks, final report",
        "handoff_to_qa_agent",
        "Fail runs with missing MD incidents, failed physics gates, or failed GraphDB ingest.",
    ),
)


def agents_sdk_available() -> bool:
    return importlib.util.find_spec("agents") is not None


def build_agents_sdk_team(endpoint: ModelProviderConfig, session_id: str, session_path: Path | None = None) -> AgentsSdkTeam:
    try:
        from agents import Agent, SQLiteSession, handoff
    except ImportError as exc:
        raise AgentsSdkRuntimeError("openai_agents_sdk_missing") from exc

    specialists = {
        role.role_id: Agent(
            name=role.display_name,
            handoff_description=role.boundary,
            instructions=role.instructions,
            model=endpoint.model,
        )
        for role in AGENT_ROLES
    }
    handoffs = [
        handoff(
            specialists[role.role_id],
            tool_name_override=role.handoff_tool_name,
            tool_description_override=role.boundary,
        )
        for role in AGENT_ROLES
    ]
    orchestrator = Agent(
        name="Orchestrator",
        handoff_description="Owns the simulation run and routes work to specialist agents.",
        instructions=(
            "Clarify missing inputs, route work to specialists, preserve approval boundaries, "
            "and stop on hard physics/data blockers."
        ),
        handoffs=handoffs,
        model=endpoint.model,
    )
    return AgentsSdkTeam(
        orchestrator=orchestrator,
        specialists=specialists,
        session=SQLiteSession(session_id, str(session_path or ensure_runtime_session_path(session_id))),
        handoff_tool_names=tuple(role.handoff_tool_name for role in AGENT_ROLES),
    )


def run_agents_sdk_fake_gateway_smoke(
    endpoint: ModelProviderConfig,
    session_id: str,
    user_goal: str,
    session_path: Path | None = None,
) -> str:
    try:
        from agents import RunConfig, Runner, Usage
        from agents.items import ModelResponse
        from agents.models.interface import Model, ModelProvider
        from openai.types.responses.response_output_message import ResponseOutputMessage
        from openai.types.responses.response_output_text import ResponseOutputText
    except ImportError as exc:
        raise AgentsSdkRuntimeError("openai_agents_sdk_missing") from exc

    class FakeGatewayModel(Model):
        async def get_response(
            self,
            system_instructions: str | None,
            input: Any,
            model_settings: Any,
            tools: list[Any],
            output_schema: Any,
            handoffs: list[Any],
            tracing: Any,
            *,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt: Any,
        ) -> Any:
            output_text = ResponseOutputText(type="output_text", text="agents_sdk_runtime_ready", annotations=[])
            message = ResponseOutputMessage(
                id="msg_agents_sdk_runtime_ready",
                type="message",
                role="assistant",
                content=[output_text],
                status="completed",
            )
            return ModelResponse(
                output=[message],
                usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
                response_id="resp_agents_sdk_runtime_ready",
            )

        def stream_response(self, *args: Any, **kwargs: Any) -> Any:
            async def _empty_stream() -> Any:
                if False:
                    yield None

            return _empty_stream()

    class FakeGatewayModelProvider(ModelProvider):
        def get_model(self, model_name: str | None) -> Any:
            return FakeGatewayModel()

    team = build_agents_sdk_team(endpoint, session_id, session_path)
    result = Runner.run_sync(
        team.orchestrator,
        user_goal,
        max_turns=1,
        session=team.session,
        run_config=RunConfig(
            model_provider=FakeGatewayModelProvider(),
            tracing_disabled=True,
            workflow_name="Atomistic Simulation Agent SDK smoke",
        ),
    )
    return str(result.final_output)


def run_agents_sdk_runtime_dry_run(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    *,
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
    skill_invocations = run_registered_agent_skills(payload)
    messages = _message_log()
    approvals = _approval_gates(payload)
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
        messages=messages,
        trace=trace,
        approval_gates=approvals,
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
