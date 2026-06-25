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

WORKFLOW_GATE_SCHEMA_VERSION: Final = "workflow_gate_v1"
SAFE_RUNTIME_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
ORCHESTRATOR_AGENT_ID: Final = "orchestrator"


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

    def to_json(self) -> JsonMap:
        payload: dict[str, object] = {
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

    def to_json(self) -> JsonMap:
        return {
            "workflow_id": self.workflow_id,
            "gate_id": self.gate_id,
            "status": self.status,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
            "answered_at": self.answered_at,
        }


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
    shape: dict[str, object] = {
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
    )


def gate_response_for_gate(
    gate: WorkflowGate, status: str, blockers: tuple[str, ...], answered_at: str
) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(
        gate.workflow_id, gate.gate_id, status, gate.owner_agent_id, gate.target_agent_id, gate.ledger_ref, blockers, answered_at
    )


def gate_response_blocked(workflow_id: str, gate_id: str, blocker: str) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(workflow_id, gate_id, "blocked", "", "", gate_ledger_ref(workflow_id, gate_id), (blocker,))


def workflow_authority_blocker(actor: str, owner: str, target: str) -> str:
    if actor == ORCHESTRATOR_AGENT_ID:
        return ""
    if target == ORCHESTRATOR_AGENT_ID:
        return "workflow_authority_orchestrator_denied"
    if actor == owner and actor == target:
        return ""
    return "workflow_authority_peer_denied"


def response_schema_blocker(value: object, schema: JsonMap) -> str:
    schema_type = schema.get("type")
    if schema_type is not None and not matches_schema_type(value, schema_type):
        return "workflow_gate_response_schema_mismatch"
    required = schema.get("required")
    if isinstance(required, list):
        if not isinstance(value, dict):
            return "workflow_gate_response_schema_mismatch"
        for field in required:
            if isinstance(field, str) and field not in value:
                return "workflow_gate_response_schema_mismatch"
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(value, dict):
        for field, field_schema in properties.items():
            if not isinstance(field, str) or field not in value or not isinstance(field_schema, dict):
                continue
            blocker = response_schema_blocker(value[field], field_schema)
            if blocker:
                return blocker
    return ""


def matches_schema_type(value: object, schema_type: object) -> bool:
    if isinstance(schema_type, list):
        return any(matches_schema_type(value, item) for item in schema_type)
    if not isinstance(schema_type, str):
        return True
    match schema_type:
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "string":
            return isinstance(value, str)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case _:
            return True


def gate_kind(value: object) -> WorkflowGateKind | None:
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
