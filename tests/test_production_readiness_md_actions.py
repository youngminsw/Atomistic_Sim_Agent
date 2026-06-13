from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from production_readiness_fixtures import actions_by_id, amorphous_blocked_ledger, string_list
from sim_agent.production_readiness import assess_production_readiness_from_payloads


def test_production_readiness_requests_amorphous_structure_prep_action() -> None:
    report = assess_production_readiness_from_payloads(amorphous_blocked_ledger())
    action_plan = actions_by_id(report.payload)

    assert "amorphous_lammps_structure_source_required" in string_list(
        report.payload, "hard_blockers"
    )
    assert "prepare_or_import_relaxed_amorphous_structure" in string_list(
        report.payload, "agent_actions"
    )
    prep_action = action_plan["prepare_or_import_relaxed_amorphous_structure"]
    assert prep_action["actor"] == "md_agent"
    assert prep_action["status"] == "requires_structure_source"
    assert string_list(prep_action, "missing_artifacts") == [
        "amorphous_lammps_structure_source"
    ]
    assert string_list(prep_action, "next_actions") == [
        "prepare_amorphous_structure_prep_worker_bundle",
        "request_qa_review_before_remote_submission",
        "run_amorphous_structure_prep_worker_after_approval",
    ]
    resolve_action = action_plan["resolve_md_production_blockers"]
    assert resolve_action["actor"] == "md_agent"
    assert resolve_action["status"] == "blocked_on_md_physics_gate"
    assert "amorphous_lammps_structure_source_required" in string_list(
        resolve_action, "hard_blockers"
    )
    assert "prepare_or_import_relaxed_amorphous_structure" in string_list(
        resolve_action, "next_actions"
    )


def test_production_readiness_uses_existing_amorphous_prep_worker() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "amorphous_structure_prep_worker_path": "/tmp/run/amorphous_structure_prep_worker_bundle.json",
        "amorphous_structure_prep_remote_plan_path": "/tmp/run/amorphous_structure_prep_remote_plan.json",
    }

    report = assess_production_readiness_from_payloads(ledger)
    action_plan = actions_by_id(report.payload)

    assert "amorphous_structure_prep_worker_bundle_written" in string_list(
        report.payload, "evidence"
    )
    assert (
        "run_amorphous_structure_prep_worker_after_approval"
        in string_list(report.payload, "agent_actions")
    )
    assert "rerun_agent_with_relaxed_amorphous_structure_source" in string_list(
        report.payload, "agent_actions"
    )
    assert "prepare_or_import_relaxed_amorphous_structure" not in string_list(
        report.payload, "agent_actions"
    )
    prep_action = action_plan["run_amorphous_structure_prep_worker_after_approval"]
    assert prep_action["status"] == "ready_after_user_action"
    assert prep_action["requires_user_action"] == "approve_remote_or_long_compute_run"
    assert string_list(prep_action, "command") == [
        "python3",
        "02.Source_code/mss_agent/scripts/run_remote_execution_plan.py",
        "--plan",
        "/tmp/run/amorphous_structure_prep_remote_plan.json",
        "--out",
        "/tmp/run/amorphous_structure_prep_remote_result.json",
    ]


def test_production_readiness_reports_failed_amorphous_prep_worker() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "amorphous_structure_prep_worker_path": "/tmp/run/amorphous_structure_prep_worker_bundle.json"
    }
    ledger["remote"] = {
        "amorphous_prep_status": "remote_plan_failed",
        "amorphous_prep_blockers": ["remote_plan_command_failed"],
        "chain_status": "",
        "chain_blockers": [],
    }

    report = assess_production_readiness_from_payloads(ledger)

    assert "remote_plan_command_failed" in string_list(report.payload, "hard_blockers")
    assert "amorphous_structure_prep_remote_failed" in string_list(
        report.payload, "evidence"
    )
    assert "diagnose_amorphous_structure_prep_remote_failure" in string_list(
        report.payload, "agent_actions"
    )
    assert (
        "run_amorphous_structure_prep_worker_after_approval"
        not in string_list(report.payload, "agent_actions")
    )


def test_production_readiness_uses_completed_amorphous_prep_worker() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "amorphous_structure_prep_worker_path": "/tmp/run/amorphous_structure_prep_worker_bundle.json",
        "amorphous_structure_source_path": "/tmp/run/amorphous_structure_prep/amorphous_structure_source.json",
        "validated_request_path": "/tmp/run/validated_request.json",
    }
    ledger["remote"] = {
        "amorphous_prep_status": "remote_plan_completed",
        "amorphous_prep_blockers": [],
        "chain_status": "",
        "chain_blockers": [],
    }

    report = assess_production_readiness_from_payloads(ledger)
    action_plan = actions_by_id(report.payload)

    assert "amorphous_structure_prep_remote_completed" in string_list(
        report.payload, "evidence"
    )
    assert "rerun_agent_with_relaxed_amorphous_structure_source" in string_list(
        report.payload, "agent_actions"
    )
    assert (
        "run_amorphous_structure_prep_worker_after_approval"
        not in string_list(report.payload, "agent_actions")
    )
    rerun_action = action_plan["rerun_agent_with_relaxed_amorphous_structure_source"]
    assert rerun_action["status"] == "ready"
    assert string_list(rerun_action, "command") == [
        "python3",
        "02.Source_code/mss_agent/scripts/resume_agent_run_from_request.py",
        "--request",
        "/tmp/run/validated_request.json",
        "--lammps-structure-source",
        "/tmp/run/amorphous_structure_prep/amorphous_structure_source.json",
        "--output-dir",
        "/tmp/run/resumed_with_amorphous_structure",
        "--host",
        "gpu-5090",
        "--environment-name",
        "atomistic-sim-gpu",
        "--ssh-target",
        "swym@10.24.12.85",
        "--ssh-port",
        "55555",
    ]
