from __future__ import annotations

from collections.abc import Callable

from sim_agent.production_readiness_contract import unknown_action_contract_error
from sim_agent.production_readiness_graphdb_actions import graphdb_apply_action_entry
from sim_agent.production_readiness_md_action_builders import (
    diagnose_amorphous_structure_prep_remote_failure_action_entry,
    prepare_or_import_amorphous_structure_action_entry,
    remote_plan_action_entry,
    resolve_md_production_blockers_action_entry,
    resume_with_structure_action_entry,
)
from sim_agent.production_readiness_missing_action import missing_action as _missing_action
from sim_agent.production_readiness_remote_action_builders import (
    model_endpoint_action_entry,
    remote_chain_action_entry,
)
from sim_agent.production_readiness_surrogate_feature_actions import (
    feature_scale_action_entry,
    surrogate_training_action_entry,
)
from sim_agent.schemas._parse import JsonMap


ActionBuilder = Callable[[JsonMap, str, list[str]], JsonMap]

ACTION_BUILDERS: dict[str, ActionBuilder] = {
    "run_amorphous_structure_prep_worker_after_approval": remote_plan_action_entry,
    "rerun_agent_with_relaxed_amorphous_structure_source": resume_with_structure_action_entry,
    "run_remote_chain_after_approval": remote_chain_action_entry,
    "run_model_endpoint_smoke_after_credentials": model_endpoint_action_entry,
    "apply_graphdb_import_after_approval": graphdb_apply_action_entry,
    "prepare_or_import_relaxed_amorphous_structure": (
        prepare_or_import_amorphous_structure_action_entry
    ),
    "diagnose_amorphous_structure_prep_remote_failure": (
        diagnose_amorphous_structure_prep_remote_failure_action_entry
    ),
    "resolve_md_production_blockers": resolve_md_production_blockers_action_entry,
    "train_or_active_learn_surrogate": surrogate_training_action_entry,
    "run_feature_scale_from_accepted_production_surrogate": feature_scale_action_entry,
}


def action_plan(
    ledger: JsonMap,
    agent_actions: list[str],
    user_actions: list[str],
) -> list[JsonMap]:
    return [action_entry(ledger, action, user_actions) for action in agent_actions]


def action_entry(ledger: JsonMap, action: str, user_actions: list[str]) -> JsonMap:
    builder = ACTION_BUILDERS.get(action)
    if builder is None:
        return unknown_action_contract_error(action)
    return builder(ledger, action, user_actions)
