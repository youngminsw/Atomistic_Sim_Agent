from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import now, required_text, safe_id, workflow_authority_blocker, write_json


WORKFLOW_GOAL_SCHEMA_VERSION = "workflow_goal_v1"
WORKFLOW_GOAL_OPERATIONS: tuple[str, ...] = ("create", "get", "resume", "pause", "drop", "complete")


@dataclass(frozen=True, slots=True)
class WorkflowGoalOperationResult:
    workflow_id: str
    goal_id: str
    operation: str
    status: str
    actor_agent_id: str
    owner_agent_id: str
    target_agent_id: str
    state: str
    ledger_ref: str
    blockers: tuple[str, ...]
    goal: JsonMap | None = None

    def to_json(self) -> JsonMap:
        payload: dict[str, str | list[str] | JsonMap] = {
            "schema_version": WORKFLOW_GOAL_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "goal_id": self.goal_id,
            "operation": self.operation,
            "status": self.status,
            "actor_agent_id": self.actor_agent_id,
            "owner_agent_id": self.owner_agent_id,
            "target_agent_id": self.target_agent_id,
            "state": self.state,
            "ledger_ref": self.ledger_ref,
            "blockers": list(self.blockers),
        }
        if self.goal is not None:
            payload["goal"] = self.goal
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowGoalRecord:
    goal: JsonMap | None
    corrupt: bool = False


def operate_workflow_goal(output_dir: Path, payload: JsonMap) -> WorkflowGoalOperationResult:
    request = _operation_request(payload)
    malformed = _request_malformed(request)
    if malformed:
        return _blocked_result(request, "workflow_goal_malformed")
    if request.operation not in WORKFLOW_GOAL_OPERATIONS:
        return _blocked_result(request, "workflow_goal_unknown_operation")
    authority_blocker = workflow_authority_blocker(request.actor, request.owner, request.target)
    if authority_blocker:
        return _blocked_result(request, authority_blocker)

    record = _read_goal(output_dir / request.ledger_ref)
    if record.corrupt:
        return _blocked_result(request, "workflow_goal_corrupt")
    existing = record.goal
    if existing is None:
        if request.operation == "create":
            return _write_goal(output_dir, payload, request, None)
        return _blocked_result(request, "workflow_goal_unknown")

    stored = _stored_goal_identity(existing)
    if stored is None or stored.workflow_id != request.workflow_id or stored.goal_id != request.goal_id:
        return _blocked_result(request, "workflow_goal_corrupt")
    stored_authority_blocker = workflow_authority_blocker(request.actor, stored.owner, stored.target)
    if stored_authority_blocker:
        return _blocked_result(request.with_owner_target(stored.owner, stored.target), stored_authority_blocker, existing)
    if request.operation == "create":
        if request.owner == stored.owner and request.target == stored.target:
            return _accepted_result(request.with_owner_target(stored.owner, stored.target), stored.state, existing)
        return _blocked_result(request.with_owner_target(stored.owner, stored.target), "workflow_goal_conflict", existing)
    if request.operation == "get":
        return _accepted_result(request.with_owner_target(stored.owner, stored.target), stored.state, existing)
    if stored.state in {"complete", "dropped"}:
        return _blocked_result(
            request.with_owner_target(stored.owner, stored.target),
            "workflow_goal_terminal",
            existing,
            stored.state,
        )
    return _write_goal(output_dir, payload, request.with_owner_target(stored.owner, stored.target), existing)


@dataclass(frozen=True, slots=True)
class WorkflowGoalRequest:
    operation: str
    actor: str
    owner: str
    target: str
    workflow_id: str
    goal_id: str
    ledger_ref: str

    def with_owner_target(self, owner: str, target: str) -> WorkflowGoalRequest:
        return WorkflowGoalRequest(self.operation, self.actor, owner, target, self.workflow_id, self.goal_id, self.ledger_ref)


@dataclass(frozen=True, slots=True)
class StoredGoalIdentity:
    workflow_id: str
    goal_id: str
    owner: str
    target: str
    state: str


def _operation_request(payload: JsonMap) -> WorkflowGoalRequest:
    workflow_id = required_text(payload, "workflow_id")
    goal_id = required_text(payload, "goal_id")
    return WorkflowGoalRequest(
        required_text(payload, "operation"),
        required_text(payload, "actor_agent_id"),
        required_text(payload, "owner_agent_id"),
        required_text(payload, "target_agent_id"),
        workflow_id,
        goal_id,
        _workflow_goal_ledger_ref(workflow_id or "unknown", goal_id or "unknown"),
    )


def _request_malformed(request: WorkflowGoalRequest) -> bool:
    return (
        not request.operation
        or not request.actor
        or not request.owner
        or not request.target
        or not request.workflow_id
        or not request.goal_id
    )


def _blocked_result(
    request: WorkflowGoalRequest,
    blocker: str,
    goal: JsonMap | None = None,
    state: str = "",
) -> WorkflowGoalOperationResult:
    return WorkflowGoalOperationResult(
        request.workflow_id,
        request.goal_id,
        request.operation,
        "blocked",
        request.actor,
        request.owner,
        request.target,
        state,
        request.ledger_ref,
        (blocker,),
        goal,
    )


def _accepted_result(request: WorkflowGoalRequest, state: str, goal: JsonMap) -> WorkflowGoalOperationResult:
    return WorkflowGoalOperationResult(
        request.workflow_id,
        request.goal_id,
        request.operation,
        "accepted",
        request.actor,
        request.owner,
        request.target,
        state,
        request.ledger_ref,
        (),
        goal,
    )


def _write_goal(
    output_dir: Path,
    payload: JsonMap,
    request: WorkflowGoalRequest,
    existing: JsonMap | None,
) -> WorkflowGoalOperationResult:
    state = _workflow_goal_state_for_operation(request.operation)
    goal = _workflow_goal_payload(payload, existing, request, state)
    write_json(output_dir / request.ledger_ref, goal)
    return _accepted_result(request, state, goal)


def _workflow_goal_ledger_ref(workflow_id: str, goal_id: str) -> str:
    return f"{safe_id(workflow_id)}/goals/{safe_id(goal_id)}.json"


def _workflow_goal_state_for_operation(operation: str) -> str:
    states = {
        "create": "active",
        "resume": "active",
        "pause": "paused",
        "drop": "dropped",
        "complete": "complete",
        "get": "",
    }
    return states.get(operation, "")


def _workflow_goal_payload(
    payload: JsonMap,
    existing: JsonMap | None,
    request: WorkflowGoalRequest,
    state: str,
) -> JsonMap:
    timestamp = now()
    history = existing.get("history") if existing is not None else None
    history_items = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    history_items.append(
        {
            "operation": request.operation,
            "state": state,
            "at": timestamp,
            "actor_agent_id": request.actor,
        }
    )
    created_at = required_text(existing or {}, "created_at") or timestamp
    objective = required_text(payload, "objective") or required_text(existing or {}, "objective")
    return {
        "schema_version": WORKFLOW_GOAL_SCHEMA_VERSION,
        "workflow_id": request.workflow_id,
        "goal_id": request.goal_id,
        "owner_agent_id": request.owner,
        "target_agent_id": request.target,
        "state": state,
        "objective": objective,
        "created_at": created_at,
        "updated_at": timestamp,
        "history": history_items,
    }


def _read_goal(path: Path) -> WorkflowGoalRecord:
    if not path.is_file():
        return WorkflowGoalRecord(None)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return WorkflowGoalRecord(None, True)
    return WorkflowGoalRecord(raw) if isinstance(raw, dict) else WorkflowGoalRecord(None, True)


def _stored_goal_identity(goal: JsonMap) -> StoredGoalIdentity | None:
    workflow_id = required_text(goal, "workflow_id")
    goal_id = required_text(goal, "goal_id")
    owner = required_text(goal, "owner_agent_id")
    target = required_text(goal, "target_agent_id")
    state = required_text(goal, "state")
    if not workflow_id or not goal_id or not owner or not target or not state:
        return None
    return StoredGoalIdentity(workflow_id, goal_id, owner, target, state)
