from __future__ import annotations

from pathlib import Path

from sim_agent.production_readiness_contract import action_actor, missing_action_recovery_steps
from sim_agent.production_readiness_ledger import (
    SCRIPT_ROOT,
    amorphous_prep_blockers,
    amorphous_prep_status,
    amorphous_prep_worker_present,
    artifact_dir,
    artifact_output,
    artifact_path,
    int_text,
    mapping,
    missing_fields,
    string_list,
    text,
)
from sim_agent.production_readiness_missing_action import missing_action
from sim_agent.schemas._parse import JsonMap


def remote_plan_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    return _remote_plan_action(
        ledger,
        action,
        "amorphous_structure_prep_remote_plan_path",
        "amorphous_structure_prep_remote_result_path",
        "amorphous_structure_prep_remote_result.json",
    )


def resume_with_structure_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    return _resume_with_structure_action(ledger, action)


def prepare_or_import_amorphous_structure_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "requires_structure_source",
        "command": [],
        "missing_artifacts": ["amorphous_lammps_structure_source"],
        "acceptable_sources": [
            "relaxed_lammps_data",
            "imported_lammps_data",
            "remote_amorphous_structure_prep_worker",
        ],
        "next_actions": missing_action_recovery_steps(action),
    }


def diagnose_amorphous_structure_prep_remote_failure_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    blockers = amorphous_prep_blockers(ledger) or ["amorphous_structure_prep_remote_failed"]
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "blocked_on_failed_remote_job",
        "command": [],
        "hard_blockers": blockers,
        "next_actions": missing_action_recovery_steps(action),
    }


def resolve_md_production_blockers_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    md = mapping(ledger, "md")
    blockers = string_list(md, "hard_blockers") or ["md_production_not_ready"]
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "blocked_on_md_physics_gate",
        "command": [],
        "hard_blockers": blockers,
        "expected_incident_count": int_text(md, "expected_incident_count") or "500",
        "next_actions": _md_resolution_next_actions(ledger, blockers),
    }


def _remote_plan_action(
    ledger: JsonMap,
    action: str,
    plan_field: str,
    result_field: str,
    default_result_name: str,
) -> JsonMap:
    actor = action_actor(action)
    plan_path = artifact_path(ledger, plan_field)
    result_path = artifact_path(ledger, result_field) or artifact_output(
        ledger,
        default_result_name,
    )
    if not plan_path:
        return missing_action(action, [plan_field], actor=actor)
    return {
        "action": action,
        "actor": actor,
        "status": "ready_after_user_action",
        "requires_user_action": "approve_remote_or_long_compute_run",
        "command": [
            "python3",
            f"{SCRIPT_ROOT}/run_remote_execution_plan.py",
            "--plan",
            plan_path,
            "--out",
            result_path,
        ],
        "expected_artifacts": [result_path],
    }


def _resume_with_structure_action(ledger: JsonMap, action: str) -> JsonMap:
    request_path = artifact_path(ledger, "validated_request_path")
    source_path = artifact_path(ledger, "amorphous_structure_source_path")
    compute_target = mapping(ledger, "compute_target")
    host = text(compute_target, "host")
    environment_name = text(compute_target, "environment_name")
    missing = missing_fields(
        (
            ("validated_request_path", request_path),
            ("amorphous_structure_source_path", source_path),
            ("compute_target.host", host),
            ("compute_target.environment_name", environment_name),
        )
    )
    if missing:
        return missing_action(action, missing)
    output_dir = str(Path(artifact_dir(ledger)) / "resumed_with_amorphous_structure")
    command = [
        "python3",
        f"{SCRIPT_ROOT}/resume_agent_run_from_request.py",
        "--request",
        request_path,
        "--lammps-structure-source",
        source_path,
        "--output-dir",
        output_dir,
        "--host",
        host,
        "--environment-name",
        environment_name,
    ]
    ssh_target = text(compute_target, "ssh_target")
    ssh_port = int_text(compute_target, "ssh_port")
    if ssh_target and ssh_port:
        command.extend(["--ssh-target", ssh_target, "--ssh-port", ssh_port])
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "ready",
        "command": command,
        "expected_artifacts": [str(Path(output_dir) / "agent_run_ledger.json")],
    }


def _md_resolution_next_actions(ledger: JsonMap, blockers: list[str]) -> list[str]:
    if "amorphous_lammps_structure_source_required" not in blockers:
        return ["inspect_md_gate_report", "repair_md_campaign_inputs", "rerun_md_campaign_after_qa"]
    match amorphous_prep_status(ledger):
        case "remote_plan_failed":
            return ["diagnose_amorphous_structure_prep_remote_failure"]
        case "remote_plan_completed":
            return ["rerun_agent_with_relaxed_amorphous_structure_source"]
        case "":
            if amorphous_prep_worker_present(ledger):
                return [
                    "run_amorphous_structure_prep_worker_after_approval",
                    "rerun_agent_with_relaxed_amorphous_structure_source",
                ]
            return ["prepare_or_import_relaxed_amorphous_structure"]
        case _:
            return ["inspect_unknown_amorphous_structure_prep_status"]
