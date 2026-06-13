from __future__ import annotations

from dataclasses import dataclass

from sim_agent.production_readiness_ledger import string_list, text
from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class ProductionActionSpec:
    actor: str
    missing_recovery_steps: tuple[str, ...]


UNKNOWN_ACTION_REPAIR_STEPS = (
    "repair_production_readiness_action_registry",
    "add_action_contract_regression_test",
)

PRODUCTION_ACTION_SPECS: dict[str, ProductionActionSpec] = {
    "run_amorphous_structure_prep_worker_after_approval": ProductionActionSpec(
        actor="md_agent",
        missing_recovery_steps=(
            "prepare_amorphous_structure_prep_worker_bundle",
            "request_qa_review_before_remote_submission",
            "run_amorphous_structure_prep_worker_after_approval",
        ),
    ),
    "rerun_agent_with_relaxed_amorphous_structure_source": ProductionActionSpec(
        actor="orchestrator",
        missing_recovery_steps=(
            "prepare_or_import_relaxed_amorphous_structure",
            "rerun_agent_with_relaxed_amorphous_structure_source",
        ),
    ),
    "run_remote_chain_after_approval": ProductionActionSpec(
        actor="orchestrator",
        missing_recovery_steps=(
            "prepare_agent_compute_bundle_with_ssh_target",
            "request_qa_review_before_remote_submission",
            "run_remote_chain_after_approval",
        ),
    ),
    "run_model_endpoint_smoke_after_credentials": ProductionActionSpec(
        actor="orchestrator",
        missing_recovery_steps=(
            "persist_validated_request_from_agent_plan",
            "run_model_endpoint_smoke_after_credentials",
        ),
    ),
    "apply_graphdb_import_after_approval": ProductionActionSpec(
        actor="research_graphdb_agent",
        missing_recovery_steps=(
            "build_research_graphdb_import_bundle",
            "request_empty_neo4j_write_approval",
            "apply_graphdb_import_after_approval",
        ),
    ),
    "prepare_or_import_relaxed_amorphous_structure": ProductionActionSpec(
        actor="md_agent",
        missing_recovery_steps=(
            "prepare_amorphous_structure_prep_worker_bundle",
            "request_qa_review_before_remote_submission",
            "run_amorphous_structure_prep_worker_after_approval",
        ),
    ),
    "diagnose_amorphous_structure_prep_remote_failure": ProductionActionSpec(
        actor="md_agent",
        missing_recovery_steps=(
            "inspect_remote_plan_result",
            "repair_worker_environment_or_job_script",
            "request_qa_review_before_resubmission",
        ),
    ),
    "resolve_md_production_blockers": ProductionActionSpec(
        actor="md_agent",
        missing_recovery_steps=(
            "inspect_md_gate_report",
            "repair_md_campaign_inputs",
            "rerun_md_campaign_after_qa",
        ),
    ),
    "train_or_active_learn_surrogate": ProductionActionSpec(
        actor="ml_mdn_agent",
        missing_recovery_steps=(
            "complete_production_md_chain",
            "postprocess_lammps_execution",
            "assess_surrogate_training_gate",
        ),
    ),
    "run_feature_scale_from_accepted_production_surrogate": ProductionActionSpec(
        actor="feature_scale_agent",
        missing_recovery_steps=(
            "train_or_active_learn_surrogate",
            "prepare_feature_scale_settings",
            "run_feature_scale_from_accepted_production_surrogate",
        ),
    ),
}


def action_plan_hard_blockers(action_plan: list[JsonMap]) -> list[str]:
    blockers: list[str] = []
    for entry in action_plan:
        blockers.extend(string_list(entry, "hard_blockers"))
        if (
            text(entry, "status") == "blocked_on_missing_artifacts"
            and not string_list(entry, "next_actions")
        ):
            action = text(entry, "action") or "unknown"
            blockers.append(f"missing_recovery_steps_for_action={action}")
    return blockers


def action_actor(action: str) -> str:
    spec = PRODUCTION_ACTION_SPECS.get(action)
    return spec.actor if spec else "orchestrator"


def missing_action_recovery_steps(action: str) -> list[str]:
    spec = PRODUCTION_ACTION_SPECS.get(action)
    if spec is None:
        return list(UNKNOWN_ACTION_REPAIR_STEPS)
    if not spec.missing_recovery_steps:
        return [
            f"define_domain_recovery_steps_for_action={action}",
            "add_action_contract_regression_test",
        ]
    return list(spec.missing_recovery_steps)


def unknown_action_contract_error(action: str) -> JsonMap:
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "blocked_on_unknown_action_contract",
        "command": [],
        "hard_blockers": [f"unknown_production_readiness_action={action}"],
        "next_actions": list(UNKNOWN_ACTION_REPAIR_STEPS),
    }
