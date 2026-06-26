from __future__ import annotations

from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import WorkflowGoalAuthorityResult, now, required_text, workflow_authority_blocker, write_json
from .workflow_goal_runtime import _read_goal, _stored_goal_identity, _workflow_goal_ledger_ref


def adjust_workflow_goal_state(output_dir: Path, payload: JsonMap) -> WorkflowGoalAuthorityResult:
    actor = required_text(payload, "actor_agent_id")
    owner = required_text(payload, "owner_agent_id")
    target = required_text(payload, "target_agent_id")
    workflow_id = required_text(payload, "workflow_id")
    goal_id = required_text(payload, "goal_id")
    state = required_text(payload, "state")
    ledger_ref = _workflow_goal_ledger_ref(workflow_id, goal_id)
    record = _read_goal(output_dir / ledger_ref)
    if record.corrupt:
        return WorkflowGoalAuthorityResult(
            workflow_id, goal_id, "blocked", actor, owner, target, state, ledger_ref, ("workflow_goal_corrupt",)
        )
    existing = record.goal
    if existing is not None:
        stored = _stored_goal_identity(existing)
        if stored is None or stored.workflow_id != workflow_id or stored.goal_id != goal_id:
            return WorkflowGoalAuthorityResult(
                workflow_id, goal_id, "blocked", actor, owner, target, state, ledger_ref, ("workflow_goal_corrupt",)
            )
        owner = stored.owner
        target = stored.target
    blocker = workflow_authority_blocker(actor, owner, target)
    status = "accepted" if blocker == "" else "blocked"
    result = WorkflowGoalAuthorityResult(
        workflow_id, goal_id, status, actor, owner, target, state, ledger_ref, () if blocker == "" else (blocker,)
    )
    if status == "accepted":
        write_json(output_dir / ledger_ref, _adjusted_goal(existing, result) if existing is not None else result.to_json())
    return result


def _adjusted_goal(existing: JsonMap, result: WorkflowGoalAuthorityResult) -> JsonMap:
    timestamp = now()
    history = existing.get("history")
    history_items = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    history_items.append(
        {
            "operation": "adjust",
            "state": result.state,
            "at": timestamp,
            "actor_agent_id": result.actor_agent_id,
        }
    )
    return dict(existing) | {"state": result.state, "updated_at": timestamp, "history": history_items}
