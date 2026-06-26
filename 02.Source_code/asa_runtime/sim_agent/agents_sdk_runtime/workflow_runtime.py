from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

from sim_agent.schemas._parse import JsonMap

from .workflow_actions import WorkflowActionResolveRequest, ensure_pending_action, resolve_workflow_action
from .workflow_artifact_validation import workflow_artifact_missing_evidence, workflow_artifact_validation_blocker
from .workflow_deep_interview import (
    deep_interview_corrupt_state,
    deep_interview_gate,
    deep_interview_handoff_refs,
    deep_interview_pending_gate,
    deep_interview_response_blocker,
    record_deep_interview_response,
)
from .workflow_ralplan import ralplan_response_blocker, record_ralplan_approval_response
from .workflow_ultragoal import ultragoal_response_blocker, record_ultragoal_signoff_response
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
    workflow_authority_blocker,
    workflow_gate_schema_hash,
    write_json,
)
from .workflow_goal_runtime import (
    WORKFLOW_GOAL_OPERATIONS,
    WORKFLOW_GOAL_SCHEMA_VERSION,
    WorkflowGoalOperationResult,
    operate_workflow_goal,
)
from .workflow_goal_authority_runtime import adjust_workflow_goal_state


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
    if request.workflow_id == "deep-interview":
        return _start_deep_interview_runtime(request)
    artifact_blocker = workflow_artifact_validation_blocker(request.output_dir, request.workflow_id, request.payload)
    if artifact_blocker:
        return WorkflowRuntimeStartResult(
            "blocked",
            "blocked",
            (artifact_blocker,),
            None,
            workflow_artifact_missing_evidence(request.output_dir, request.workflow_id, request.payload)
            if artifact_blocker == "ralplan_artifact_missing"
            else (),
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
    ensure_pending_action(request.output_dir, active_gate)
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
    value = payload["value"]
    idempotency_key = required_text(payload, "idempotency_key")
    if gate.status == "accepted":
        if idempotency_key:
            action_result = resolve_workflow_action(
                WorkflowActionResolveRequest(output_dir, gate, responder_agent_id, value, idempotency_key)
            )
            if action_result.status == "duplicate":
                return gate_response_for_gate(gate, "accepted", (), gate.answered_at, action_result.to_json())
            if action_result.blockers == ("workflow_action_idempotency_conflict",):
                return gate_response_for_gate(gate, "blocked", action_result.blockers, gate.answered_at, action_result.to_json())
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
    match gate.gate_kind:
        case WorkflowGateKind.ENUM:
            if not isinstance(value, str) or value not in gate.allowed_values:
                return gate_response_for_gate(gate, "blocked", ("workflow_gate_invalid_enum_value",), "")
        case WorkflowGateKind.RESPONSE_SCHEMA:
            deep_blocker = deep_interview_response_blocker(gate, value)
            if deep_blocker:
                return gate_response_for_gate(gate, "blocked", (deep_blocker,), "")
            ralplan_blocker = ralplan_response_blocker(gate, value)
            if ralplan_blocker:
                return gate_response_for_gate(gate, "blocked", (ralplan_blocker,), "")
            ultragoal_blocker = ultragoal_response_blocker(gate, value)
            if ultragoal_blocker:
                return gate_response_for_gate(gate, "blocked", (ultragoal_blocker,), "")
            blocker = response_schema_blocker(value, gate.response_schema or {})
            if blocker:
                return gate_response_for_gate(gate, "blocked", (blocker,), "")
        case unreachable:
            assert_never(unreachable)
    action_result = resolve_workflow_action(
        WorkflowActionResolveRequest(output_dir, gate, responder_agent_id, value, idempotency_key)
    )
    if action_result.status == "blocked":
        return gate_response_for_gate(gate, "blocked", action_result.blockers, "", action_result.to_json())
    answered = answered_gate(gate)
    write_json(gate_path, answered.to_json() | {"response_value": value, "responder_agent_id": responder_agent_id})
    record_deep_interview_response(output_dir, answered, value)
    record_ralplan_approval_response(output_dir, answered, value)
    record_ultragoal_signoff_response(output_dir, answered, value)
    return gate_response_for_gate(answered, "accepted", (), answered.answered_at, action_result.to_json())


def _start_deep_interview_runtime(request: WorkflowRuntimeStartRequest) -> WorkflowRuntimeStartResult:
    workflow_dir = request.output_dir / "deep-interview"
    if deep_interview_corrupt_state(workflow_dir):
        return WorkflowRuntimeStartResult("blocked", "blocked", ("deep_interview_state_corrupt",), None)
    if deep_interview_handoff_refs(workflow_dir):
        return WorkflowRuntimeStartResult("ready", "passed", (), None)
    pending = deep_interview_pending_gate(workflow_dir)
    if pending is not None:
        return WorkflowRuntimeStartResult("blocked", "awaiting_response", ("workflow_gate_response_required",), pending)
    gate = deep_interview_gate(
        request.workflow_id,
        request.goal_id,
        request.owner_agent_id,
        request.target_agent_id,
        request.payload,
    )
    existing = read_gate(request.output_dir / gate.ledger_ref)
    if existing is not None:
        if existing.status == "accepted":
            return WorkflowRuntimeStartResult(
                "blocked",
                "accepted",
                ("deep_interview_next_round_required",),
                existing,
            )
        return WorkflowRuntimeStartResult(
            "blocked",
            existing.status or "blocked",
            ("workflow_gate_response_required",),
            existing,
        )
    gate_path = request.output_dir / gate.ledger_ref
    write_json(gate_path, gate.to_json())
    ensure_pending_action(request.output_dir, gate)
    return WorkflowRuntimeStartResult("blocked", "awaiting_response", ("workflow_gate_response_required",), gate)


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
