from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

from sim_agent.schemas._parse import JsonMap

from .workflow_authority import ORCHESTRATOR_AGENT_ID, response_schema_blocker, workflow_authority_blocker

WORKFLOW_GATE_SCHEMA_VERSION: Final = "workflow_gate_v1"
SAFE_RUNTIME_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

type WorkflowGateJsonValue = str | int | float | bool | None | JsonMap | list["WorkflowGateJsonValue"] | tuple[
    "WorkflowGateJsonValue", ...
]


class WorkflowGateKind(StrEnum):
    ENUM = "enum"
    RESPONSE_SCHEMA = "response_schema"


@dataclass(frozen=True, slots=True)
class WorkflowGate:
    workflow_id: str
    goal_id: str
    gate_id: str
    gate_kind: WorkflowGateKind
    owner_agent_id: str
    target_agent_id: str
    status: str
    created_at: str
    ledger_ref: str
    blockers: tuple[str, ...]
    allowed_values: tuple[str, ...] = ()
    response_schema: JsonMap | None = None
    answered_at: str = ""
    schema_hash: str = ""
    deep_interview: JsonMap | None = None

    def to_json(self) -> JsonMap:
        payload: dict[str, WorkflowGateJsonValue] = {
            "schema_version": WORKFLOW_GATE_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "goal_id": self.goal_id,
            "gate_id": self.gate_id,
            "gate_kind": self.gate_kind.value,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "answered_at": self.answered_at,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
            "schema_hash": self.schema_hash or workflow_gate_schema_hash(self),
        }
        match self.gate_kind:
            case WorkflowGateKind.ENUM:
                payload["allowed_values"] = list(self.allowed_values)
            case WorkflowGateKind.RESPONSE_SCHEMA:
                payload["response_schema"] = self.response_schema or {}
            case unreachable:
                assert_never(unreachable)
        if self.deep_interview is not None:
            payload["deep_interview"] = dict(self.deep_interview)
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowGateResponseResult:
    workflow_id: str
    gate_id: str
    status: str
    owner_agent_id: str
    target_agent_id: str
    ledger_ref: str
    blockers: tuple[str, ...]
    answered_at: str = ""
    action_lifecycle: JsonMap | None = None

    def to_json(self) -> JsonMap:
        payload: dict[str, WorkflowGateJsonValue] = {
            "workflow_id": self.workflow_id,
            "gate_id": self.gate_id,
            "status": self.status,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
            "answered_at": self.answered_at,
        }
        if self.action_lifecycle is not None:
            payload["action_lifecycle"] = self.action_lifecycle
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowGoalAuthorityResult:
    workflow_id: str
    goal_id: str
    status: str
    actor_agent_id: str
    owner_agent_id: str
    target_agent_id: str
    state: str
    ledger_ref: str
    blockers: tuple[str, ...]

    def to_json(self) -> JsonMap:
        return {
            "workflow_id": self.workflow_id,
            "goal_id": self.goal_id,
            "status": self.status,
            "actor_agent_id": self.actor_agent_id,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "state": self.state,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
        }


def workflow_gate_schema_hash(gate: WorkflowGate) -> str:
    shape: dict[str, WorkflowGateJsonValue] = {
        "schema_version": WORKFLOW_GATE_SCHEMA_VERSION,
        "workflow_id": gate.workflow_id,
        "goal_id": gate.goal_id,
        "gate_id": gate.gate_id,
        "gate_kind": gate.gate_kind.value,
        "owner_agent_id": gate.owner_agent_id,
        "target_agent_id": gate.target_agent_id,
    }
    match gate.gate_kind:
        case WorkflowGateKind.ENUM:
            shape["allowed_values"] = list(gate.allowed_values)
        case WorkflowGateKind.RESPONSE_SCHEMA:
            shape["response_schema"] = gate.response_schema or {}
        case unreachable:
            assert_never(unreachable)
    if gate.deep_interview is not None:
        shape["deep_interview"] = dict(gate.deep_interview)
    encoded = json.dumps(shape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def read_gate(path: Path) -> WorkflowGate | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    kind = gate_kind(raw.get("gate_kind"))
    if kind is None:
        return None
    return WorkflowGate(
        required_text(raw, "workflow_id"),
        required_text(raw, "goal_id"),
        required_text(raw, "gate_id"),
        kind,
        required_text(raw, "owner_agent_id"),
        required_text(raw, "target_agent_id"),
        required_text(raw, "status"),
        required_text(raw, "created_at"),
        required_text(raw, "ledger_ref"),
        tuple(item for item in raw.get("blockers", []) if isinstance(item, str)),
        tuple(item for item in raw.get("allowed_values", []) if isinstance(item, str)),
        raw.get("response_schema") if isinstance(raw.get("response_schema"), dict) else None,
        required_text(raw, "answered_at"),
        required_text(raw, "schema_hash"),
        raw.get("deep_interview") if isinstance(raw.get("deep_interview"), dict) else None,
    )


def answered_gate(gate: WorkflowGate) -> WorkflowGate:
    return WorkflowGate(
        gate.workflow_id,
        gate.goal_id,
        gate.gate_id,
        gate.gate_kind,
        gate.owner_agent_id,
        gate.target_agent_id,
        "accepted",
        gate.created_at,
        gate.ledger_ref,
        (),
        gate.allowed_values,
        gate.response_schema,
        now(),
        gate.schema_hash,
        gate.deep_interview,
    )


def gate_response_for_gate(
    gate: WorkflowGate,
    status: str,
    blockers: tuple[str, ...],
    answered_at: str,
    action_lifecycle: JsonMap | None = None,
) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(
        gate.workflow_id,
        gate.gate_id,
        status,
        gate.owner_agent_id,
        gate.target_agent_id,
        gate.ledger_ref,
        blockers,
        answered_at,
        action_lifecycle,
    )


def gate_response_blocked(workflow_id: str, gate_id: str, blocker: str) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(workflow_id, gate_id, "blocked", "", "", gate_ledger_ref(workflow_id, gate_id), (blocker,))


def gate_kind(value: WorkflowGateJsonValue) -> WorkflowGateKind | None:
    if not isinstance(value, str):
        return None
    try:
        return WorkflowGateKind(value)
    except ValueError:
        return None


def allowed_values(payload: JsonMap) -> tuple[str, ...]:
    values = payload.get("allowed_values")
    if not isinstance(values, list | tuple):
        return ()
    return tuple(item for item in values if isinstance(item, str) and item)


def required_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return ""


def gate_ledger_ref(workflow_id: str, gate_id: str) -> str:
    return f"{safe_id(workflow_id)}/gates/{safe_id(gate_id)}.json"


def safe_id(value: str) -> str:
    return value if SAFE_RUNTIME_ID_RE.fullmatch(value) else "unknown"


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
