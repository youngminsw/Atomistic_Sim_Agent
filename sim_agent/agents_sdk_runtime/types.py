from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sim_agent.schemas._parse import JsonMap


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True)
class AgentRoleDefinition:
    role_id: str
    display_name: str
    boundary: str
    handoff_tool_name: str
    instructions: str


@dataclass(frozen=True, slots=True)
class RuntimeMessage:
    sender: str
    recipient: str
    message: str


@dataclass(frozen=True, slots=True)
class RuntimeTraceEvent:
    event_type: str
    agent: str
    summary: str


@dataclass(frozen=True, slots=True)
class ApprovalGate:
    gate_id: str
    status: ApprovalStatus
    reason: str


@dataclass(frozen=True, slots=True)
class SkillInvocationResult:
    agent_id: str
    skill_id: str
    status: str
    execution_status: str
    domain_adapter: str
    artifact_ref: str
    contract: JsonMap
    result: JsonMap


@dataclass(frozen=True, slots=True)
class AgentsSdkTeam:
    orchestrator: Any
    specialists: dict[str, Any]
    session: Any
    handoff_tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentsSdkRuntimeResult:
    run_id: str
    session_id: str
    sdk_available: bool
    sdk_run_completed: bool
    provider: str
    model: str
    reasoning_effort: str
    auth_mode: str
    session_path: str
    handoff_sequence: tuple[str, ...]
    agent_model_assignments: JsonMap
    messages: tuple[RuntimeMessage, ...]
    trace: tuple[RuntimeTraceEvent, ...]
    approval_gates: tuple[ApprovalGate, ...]
    graph_memory: JsonMap
    skill_registry: JsonMap
    skill_invocations: tuple[SkillInvocationResult, ...]
    final_output: str
