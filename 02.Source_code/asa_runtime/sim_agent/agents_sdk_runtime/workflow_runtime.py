from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
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
class WorkflowRuntimeStartRequest:
    output_dir: Path
    workflow_id: str
    actor_agent_id: str
    owner_agent_id: str
    target_agent_id: str
    goal_id: str
    payload: JsonMap
    gate_payload: JsonMap | None


@dataclass(frozen=True, slots=True)
class WorkflowRuntimeStartResult:
    status: str
    gate_status: str
    blockers: tuple[str, ...]
    gate: WorkflowGate | None
    missing_evidence: tuple[str, ...] = ()


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


def start_workflow_runtime(request: WorkflowRuntimeStartRequest) -> WorkflowRuntimeStartResult:
    authority_blocker = workflow_authority_blocker(
        request.actor_agent_id,
        request.owner_agent_id,
        request.target_agent_id,
    )
    if authority_blocker:
        return WorkflowRuntimeStartResult("blocked", "blocked", (authority_blocker,), None)
    artifact_blocker = _artifact_validation_blocker(request)
    if artifact_blocker:
        return WorkflowRuntimeStartResult(
            "blocked",
            "blocked",
            (artifact_blocker,),
            None,
            _artifact_missing_evidence(request) if artifact_blocker == "ralplan_artifact_missing" else (),
        )
    if request.gate_payload is None:
        return WorkflowRuntimeStartResult("ready", "passed", (), None)
    gate = _gate_from_payload(request)
    if gate is None:
        return WorkflowRuntimeStartResult("blocked", "blocked", ("workflow_gate_malformed",), None)
    gate_path = request.output_dir / gate.ledger_ref
    existing = _read_gate(gate_path)
    active_gate = existing or gate
    if active_gate.status == "accepted":
        return WorkflowRuntimeStartResult("ready", "passed", (), active_gate)
    _write_json(gate_path, active_gate.to_json())
    return WorkflowRuntimeStartResult("blocked", "awaiting_response", ("workflow_gate_response_required",), active_gate)


def respond_workflow_gate(output_dir: Path, payload: JsonMap) -> WorkflowGateResponseResult:
    workflow_id = _required_text(payload, "workflow_id")
    gate_id = _required_text(payload, "gate_id")
    responder_agent_id = _required_text(payload, "responder_agent_id")
    if not workflow_id or not gate_id or not responder_agent_id or "value" not in payload:
        return _gate_response_blocked(workflow_id, gate_id, "workflow_gate_malformed_response")
    gate_path = output_dir / _gate_ledger_ref(workflow_id, gate_id)
    gate = _read_gate(gate_path)
    if gate is None:
        return _gate_response_blocked(workflow_id, gate_id, "workflow_gate_unknown")
    if responder_agent_id != gate.target_agent_id:
        return _gate_response_for_gate(gate, "blocked", ("workflow_gate_responder_denied",), "")
    if gate.status == "accepted":
        return WorkflowGateResponseResult(
            gate.workflow_id,
            gate.gate_id,
            "accepted",
            gate.owner_agent_id,
            gate.target_agent_id,
            gate.ledger_ref,
            ("workflow_gate_already_answered",),
            gate.answered_at,
        )
    value = payload["value"]
    match gate.gate_kind:
        case WorkflowGateKind.ENUM:
            if not isinstance(value, str) or value not in gate.allowed_values:
                return _gate_response_for_gate(gate, "blocked", ("workflow_gate_invalid_enum_value",), "")
        case WorkflowGateKind.RESPONSE_SCHEMA:
            if value is None:
                return _gate_response_for_gate(gate, "blocked", ("workflow_gate_malformed_response",), "")
        case unreachable:
            assert_never(unreachable)
    answered_gate = _answered_gate(gate)
    _write_json(gate_path, answered_gate.to_json() | {"response_value": value, "responder_agent_id": responder_agent_id})
    return _gate_response_for_gate(answered_gate, "accepted", (), answered_gate.answered_at)


def adjust_workflow_goal_state(output_dir: Path, payload: JsonMap) -> WorkflowGoalAuthorityResult:
    actor = _required_text(payload, "actor_agent_id")
    owner = _required_text(payload, "owner_agent_id")
    target = _required_text(payload, "target_agent_id")
    workflow_id = _required_text(payload, "workflow_id")
    goal_id = _required_text(payload, "goal_id")
    state = _required_text(payload, "state")
    blocker = workflow_authority_blocker(actor, owner, target)
    status = "accepted" if blocker == "" else "blocked"
    ledger_ref = f"{_safe_id(workflow_id)}/goals/{_safe_id(goal_id)}.json"
    result = WorkflowGoalAuthorityResult(
        workflow_id, goal_id, status, actor, owner, target, state, ledger_ref, () if blocker == "" else (blocker,)
    )
    _write_json(output_dir / ledger_ref, result.to_json())
    return result


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


def _gate_from_payload(request: WorkflowRuntimeStartRequest) -> WorkflowGate | None:
    payload = request.gate_payload or {}
    gate_id = _required_text(payload, "gate_id")
    kind = _gate_kind(payload.get("gate_kind"))
    if not gate_id or kind is None:
        return None
    allowed_values = _allowed_values(payload)
    response_schema = payload.get("response_schema") if isinstance(payload.get("response_schema"), dict) else None
    if kind == WorkflowGateKind.ENUM and not allowed_values:
        return None
    if kind == WorkflowGateKind.RESPONSE_SCHEMA and response_schema is None:
        return None
    gate = WorkflowGate(
        request.workflow_id,
        request.goal_id,
        gate_id,
        kind,
        request.owner_agent_id,
        request.target_agent_id,
        "awaiting_response",
        _now(),
        _gate_ledger_ref(request.workflow_id, gate_id),
        (),
        allowed_values,
        response_schema,
    )
    return replace(gate, schema_hash=workflow_gate_schema_hash(gate))


def _read_gate(path: Path) -> WorkflowGate | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    kind = _gate_kind(raw.get("gate_kind"))
    if kind is None:
        return None
    return WorkflowGate(
        _required_text(raw, "workflow_id"),
        _required_text(raw, "goal_id"),
        _required_text(raw, "gate_id"),
        kind,
        _required_text(raw, "owner_agent_id"),
        _required_text(raw, "target_agent_id"),
        _required_text(raw, "status"),
        _required_text(raw, "created_at"),
        _required_text(raw, "ledger_ref"),
        tuple(item for item in raw.get("blockers", []) if isinstance(item, str)),
        tuple(item for item in raw.get("allowed_values", []) if isinstance(item, str)),
        raw.get("response_schema") if isinstance(raw.get("response_schema"), dict) else None,
        _required_text(raw, "answered_at"),
        _required_text(raw, "schema_hash"),
    )


def _answered_gate(gate: WorkflowGate) -> WorkflowGate:
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
        _now(),
        gate.schema_hash,
    )


def _gate_response_for_gate(
    gate: WorkflowGate, status: str, blockers: tuple[str, ...], answered_at: str
) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(
        gate.workflow_id, gate.gate_id, status, gate.owner_agent_id, gate.target_agent_id, gate.ledger_ref, blockers, answered_at
    )


def _gate_response_blocked(workflow_id: str, gate_id: str, blocker: str) -> WorkflowGateResponseResult:
    return WorkflowGateResponseResult(workflow_id, gate_id, "blocked", "", "", _gate_ledger_ref(workflow_id, gate_id), (blocker,))


def workflow_authority_blocker(actor: str, owner: str, target: str) -> str:
    if actor == ORCHESTRATOR_AGENT_ID:
        return ""
    if target == ORCHESTRATOR_AGENT_ID:
        return "workflow_authority_orchestrator_denied"
    if actor == owner and actor == target:
        return ""
    return "workflow_authority_peer_denied"


def _artifact_validation_blocker(request: WorkflowRuntimeStartRequest) -> str:
    if request.workflow_id == "ralplan" and request.payload.get("validate_artifact_paths") is True:
        evidence = request.payload.get("evidence")
        if not isinstance(evidence, dict):
            return "ralplan_artifact_missing"
        for field in ("prd_path", "test_spec_path"):
            path = _artifact_path(request.output_dir, request.payload, evidence.get(field))
            if path is None or not path.is_file():
                return "ralplan_artifact_missing"
    if request.workflow_id == "ultragoal":
        goals_path = request.payload.get("goals_path") or request.payload.get("ultragoal_goals_path")
        if goals_path is not None:
            path = _artifact_path(request.output_dir, request.payload, goals_path)
            if path is None or not path.is_file():
                return "ultragoal_goals_missing"
            try:
                goals_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return "ultragoal_goals_corrupt"
            if not isinstance(goals_payload, dict) or not isinstance(goals_payload.get("goals"), list):
                return "ultragoal_goals_corrupt"
    return ""


def _artifact_missing_evidence(request: WorkflowRuntimeStartRequest) -> tuple[str, ...]:
    if request.workflow_id != "ralplan" or request.payload.get("validate_artifact_paths") is not True:
        return ()
    evidence = request.payload.get("evidence")
    if not isinstance(evidence, dict):
        return ("prd_path", "test_spec_path")
    missing: list[str] = []
    for field in ("prd_path", "test_spec_path"):
        path = _artifact_path(request.output_dir, request.payload, evidence.get(field))
        if path is None or not path.is_file():
            missing.append(field)
    return tuple(missing)


def _artifact_path(output_dir: Path, payload: JsonMap, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    artifact_root = payload.get("artifact_root")
    base = Path(artifact_root) if isinstance(artifact_root, str) and artifact_root else output_dir
    return base / path


def _gate_kind(value: object) -> WorkflowGateKind | None:
    if not isinstance(value, str):
        return None
    try:
        return WorkflowGateKind(value)
    except ValueError:
        return None


def _allowed_values(payload: JsonMap) -> tuple[str, ...]:
    values = payload.get("allowed_values")
    if not isinstance(values, list | tuple):
        return ()
    return tuple(item for item in values if isinstance(item, str) and item)


def _required_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return ""


def _gate_ledger_ref(workflow_id: str, gate_id: str) -> str:
    return f"{_safe_id(workflow_id)}/gates/{_safe_id(gate_id)}.json"


def _safe_id(value: str) -> str:
    return value if SAFE_RUNTIME_ID_RE.fullmatch(value) else "unknown"


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
