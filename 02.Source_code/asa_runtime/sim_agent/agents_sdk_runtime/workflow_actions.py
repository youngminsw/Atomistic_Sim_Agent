from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import WorkflowGate, now, required_text, safe_id, write_json


WORKFLOW_ACTION_SCHEMA_VERSION: Final = "workflow_action_v1"


@dataclass(frozen=True, slots=True)
class WorkflowAction:
    workflow_id: str
    action_id: str
    gate_id: str
    owner_agent_id: str
    target_agent_id: str
    status: str
    created_at: str
    ledger_ref: str
    gate_ledger_ref: str
    resolver_available: bool = True
    repliable: bool = True
    resolved_at: str = ""
    resolution: JsonMap | None = None

    def to_json(self) -> JsonMap:
        payload: dict[str, object] = {
            "schema_version": WORKFLOW_ACTION_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "action_id": self.action_id,
            "gate_id": self.gate_id,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "ledger_ref": self.ledger_ref,
            "gate_ledger_ref": self.gate_ledger_ref,
            "resolver_available": self.resolver_available,
            "repliable": self.repliable,
        }
        if self.resolution is not None:
            payload["resolution"] = self.resolution
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowActionResolveRequest:
    output_dir: Path
    gate: WorkflowGate
    responder_agent_id: str
    value: object
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class WorkflowActionResolveResult:
    status: str
    blockers: tuple[str, ...]
    ledger_ref: str
    resolved_at: str = ""
    idempotency_key: str = ""

    def to_json(self) -> JsonMap:
        payload: dict[str, object] = {
            "schema_version": WORKFLOW_ACTION_SCHEMA_VERSION,
            "status": self.status,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
            "resolved_at": self.resolved_at,
        }
        if self.idempotency_key:
            payload["idempotency_key"] = self.idempotency_key
        return payload


def action_ledger_ref(workflow_id: str, action_id: str) -> str:
    return f"{safe_id(workflow_id)}/actions/{safe_id(action_id)}.json"


def ensure_pending_action(output_dir: Path, gate: WorkflowGate) -> WorkflowAction:
    path = output_dir / action_ledger_ref(gate.workflow_id, gate.gate_id)
    existing = read_action(path)
    if existing is not None:
        return existing
    action = WorkflowAction(
        gate.workflow_id,
        gate.gate_id,
        gate.gate_id,
        gate.owner_agent_id,
        gate.target_agent_id,
        "pending",
        gate.created_at,
        action_ledger_ref(gate.workflow_id, gate.gate_id),
        gate.ledger_ref,
    )
    write_json(path, action.to_json())
    return action


def resolve_workflow_action(request: WorkflowActionResolveRequest) -> WorkflowActionResolveResult:
    path = request.output_dir / action_ledger_ref(request.gate.workflow_id, request.gate.gate_id)
    action = read_action(path)
    if action is None:
        return _blocked(request.gate.workflow_id, request.gate.gate_id, "workflow_action_unknown")
    if not action.resolver_available:
        return _blocked(action.workflow_id, action.action_id, "workflow_action_resolver_unavailable")
    if not action.repliable:
        return _blocked(action.workflow_id, action.action_id, "workflow_action_non_repliable")
    match action.status:
        case "pending":
            resolved_at = now()
            resolved = WorkflowAction(
                action.workflow_id,
                action.action_id,
                action.gate_id,
                action.owner_agent_id,
                action.target_agent_id,
                "resolved",
                action.created_at,
                action.ledger_ref,
                action.gate_ledger_ref,
                action.resolver_available,
                action.repliable,
                resolved_at,
                _resolution_payload(request, resolved_at),
            )
            write_json(path, resolved.to_json())
            return WorkflowActionResolveResult("resolved", (), action.ledger_ref, resolved_at, request.idempotency_key)
        case "resolved":
            return _resolved_retry(action, request)
        case _:
            return _blocked(action.workflow_id, action.action_id, "workflow_action_unknown")


def read_action(path: Path) -> WorkflowAction | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != WORKFLOW_ACTION_SCHEMA_VERSION:
        return None
    resolution = raw.get("resolution") if isinstance(raw.get("resolution"), dict) else None
    return WorkflowAction(
        required_text(raw, "workflow_id"),
        required_text(raw, "action_id"),
        required_text(raw, "gate_id"),
        required_text(raw, "owner_agent_id"),
        required_text(raw, "target_agent_id"),
        required_text(raw, "status"),
        required_text(raw, "created_at"),
        required_text(raw, "ledger_ref"),
        required_text(raw, "gate_ledger_ref"),
        raw.get("resolver_available") is not False,
        raw.get("repliable") is not False,
        required_text(raw, "resolved_at"),
        resolution,
    )


def _resolved_retry(action: WorkflowAction, request: WorkflowActionResolveRequest) -> WorkflowActionResolveResult:
    resolution = action.resolution or {}
    stored_key = required_text(resolution, "idempotency_key")
    if request.idempotency_key and stored_key == request.idempotency_key:
        if _same_json_value(resolution.get("value"), request.value):
            return WorkflowActionResolveResult(
                "duplicate",
                (),
                action.ledger_ref,
                action.resolved_at,
                request.idempotency_key,
            )
        return WorkflowActionResolveResult(
            "blocked",
            ("workflow_action_idempotency_conflict",),
            action.ledger_ref,
            action.resolved_at,
            request.idempotency_key,
        )
    return WorkflowActionResolveResult(
        "blocked",
        ("workflow_gate_already_answered",),
        action.ledger_ref,
        action.resolved_at,
        request.idempotency_key,
    )


def _blocked(workflow_id: str, action_id: str, blocker: str) -> WorkflowActionResolveResult:
    return WorkflowActionResolveResult("blocked", (blocker,), action_ledger_ref(workflow_id, action_id))


def _resolution_payload(request: WorkflowActionResolveRequest, resolved_at: str) -> JsonMap:
    payload: dict[str, object] = {
        "responder_agent_id": request.responder_agent_id,
        "value": request.value,
        "resolved_at": resolved_at,
    }
    if request.idempotency_key:
        payload["idempotency_key"] = request.idempotency_key
    return payload


def _same_json_value(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
    )
