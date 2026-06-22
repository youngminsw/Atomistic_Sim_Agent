from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from sim_agent.schemas._parse import JsonMap


RUNTIME_EVENT_SCHEMA_VERSION: Final = "asa.runtime_event.v1"


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"


class RuntimeEventType(StrEnum):
    MODEL_START = "model_start"
    MODEL_DELTA = "model_delta"
    MODEL_END = "model_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    MESSAGE_APPEND = "message_append"
    SUBAGENT_STATUS = "subagent_status"
    COMPACTION_CHECKPOINT = "compaction_checkpoint"
    WORKFLOW_GATE = "workflow_gate"
    BLOCKER = "blocker"
    RESUME = "resume"
    CANCELLATION = "cancellation"


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
class RuntimeEvent:
    sequence: int
    at: float
    session_id: str
    turn_id: str
    event_type: RuntimeEventType
    agent_id: str
    payload: JsonMap
    correlation_id: str = ""
    parent_id: str = ""

    def to_json(self) -> JsonMap:
        event: dict[str, object] = {
            "schema_version": RUNTIME_EVENT_SCHEMA_VERSION,
            "sequence": self.sequence,
            "at": self.at,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "payload": self.payload,
        }
        if self.correlation_id:
            event["correlation_id"] = self.correlation_id
        if self.parent_id:
            event["parent_id"] = self.parent_id
        return event


@dataclass(frozen=True, slots=True)
class RuntimeEventProjection:
    sequence: int
    event_type: RuntimeEventType
    agent_id: str
    status: str
    label: str
    detail: str
    tone: str


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
