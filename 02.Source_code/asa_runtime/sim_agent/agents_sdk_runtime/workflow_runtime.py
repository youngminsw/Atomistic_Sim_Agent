from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import (
    WORKFLOW_GATE_SCHEMA_VERSION,
    WorkflowGate,
    WorkflowGateKind,
    WorkflowGateResponseResult,
    WorkflowGoalAuthorityResult,
    allowed_values,
    answered_gate,
    gate_kind,
    gate_ledger_ref,
    gate_response_blocked,
    gate_response_for_gate,
    now,
    read_gate,
    required_text,
    response_schema_blocker,
    safe_id,
    workflow_authority_blocker,
    workflow_gate_schema_hash,
    write_json,
)


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
    existing = read_gate(gate_path)
    active_gate = existing or gate
    if active_gate.status == "accepted":
        return WorkflowRuntimeStartResult("ready", "passed", (), active_gate)
    write_json(gate_path, active_gate.to_json())
    return WorkflowRuntimeStartResult("blocked", "awaiting_response", ("workflow_gate_response_required",), active_gate)


def respond_workflow_gate(output_dir: Path, payload: JsonMap) -> WorkflowGateResponseResult:
    workflow_id = required_text(payload, "workflow_id")
    gate_id = required_text(payload, "gate_id")
    responder_agent_id = required_text(payload, "responder_agent_id")
    if not workflow_id or not gate_id or not responder_agent_id or "value" not in payload:
        return gate_response_blocked(workflow_id, gate_id, "workflow_gate_malformed_response")
    gate_path = output_dir / gate_ledger_ref(workflow_id, gate_id)
    gate = read_gate(gate_path)
    if gate is None:
        return gate_response_blocked(workflow_id, gate_id, "workflow_gate_unknown")
    if responder_agent_id != gate.target_agent_id:
        return gate_response_for_gate(gate, "blocked", ("workflow_gate_responder_denied",), "")
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
                return gate_response_for_gate(gate, "blocked", ("workflow_gate_invalid_enum_value",), "")
        case WorkflowGateKind.RESPONSE_SCHEMA:
            blocker = response_schema_blocker(value, gate.response_schema or {})
            if blocker:
                return gate_response_for_gate(gate, "blocked", (blocker,), "")
        case unreachable:
            assert_never(unreachable)
    answered = answered_gate(gate)
    write_json(gate_path, answered.to_json() | {"response_value": value, "responder_agent_id": responder_agent_id})
    return gate_response_for_gate(answered, "accepted", (), answered.answered_at)


def adjust_workflow_goal_state(output_dir: Path, payload: JsonMap) -> WorkflowGoalAuthorityResult:
    actor = required_text(payload, "actor_agent_id")
    owner = required_text(payload, "owner_agent_id")
    target = required_text(payload, "target_agent_id")
    workflow_id = required_text(payload, "workflow_id")
    goal_id = required_text(payload, "goal_id")
    state = required_text(payload, "state")
    blocker = workflow_authority_blocker(actor, owner, target)
    status = "accepted" if blocker == "" else "blocked"
    ledger_ref = f"{safe_id(workflow_id)}/goals/{safe_id(goal_id)}.json"
    result = WorkflowGoalAuthorityResult(
        workflow_id, goal_id, status, actor, owner, target, state, ledger_ref, () if blocker == "" else (blocker,)
    )
    write_json(output_dir / ledger_ref, result.to_json())
    return result


def _gate_from_payload(request: WorkflowRuntimeStartRequest) -> WorkflowGate | None:
    payload = request.gate_payload or {}
    gate_id = required_text(payload, "gate_id")
    kind = gate_kind(payload.get("gate_kind"))
    if not gate_id or kind is None:
        return None
    values = allowed_values(payload)
    response_schema = payload.get("response_schema") if isinstance(payload.get("response_schema"), dict) else None
    if kind == WorkflowGateKind.ENUM and not values:
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
        now(),
        gate_ledger_ref(request.workflow_id, gate_id),
        (),
        values,
        response_schema,
    )
    return replace(gate, schema_hash=workflow_gate_schema_hash(gate))


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
