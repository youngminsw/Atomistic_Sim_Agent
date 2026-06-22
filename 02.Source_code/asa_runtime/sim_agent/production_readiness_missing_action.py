from __future__ import annotations

from sim_agent.production_readiness_contract import action_actor, missing_action_recovery_steps
from sim_agent.schemas._parse import JsonMap


def missing_action(
    action: str,
    missing_artifacts: list[str],
    *,
    actor: str | None = None,
    next_actions: list[str] | None = None,
) -> JsonMap:
    payload: dict[str, object] = {
        "action": action,
        "actor": actor or action_actor(action),
        "status": "blocked_on_missing_artifacts",
        "command": [],
        "missing_artifacts": missing_artifacts,
    }
    recovery_steps = next_actions if next_actions is not None else missing_action_recovery_steps(action)
    if recovery_steps:
        payload["next_actions"] = recovery_steps
    return payload
