from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from production_readiness_fixtures import actions_by_id, amorphous_blocked_ledger, string_list
from sim_agent.production_readiness import assess_production_readiness_from_payloads


def test_production_readiness_requests_endpoint_and_graphdb_actions() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "validated_request_path": "/tmp/run/validated_request.json",
        "remote_execution_manifest_path": "/tmp/run/remote_chain_manifest.json",
    }
    report = assess_production_readiness_from_payloads(ledger)
    action_plan = actions_by_id(report.payload)

    assert "run_model_endpoint_smoke_after_credentials" in string_list(
        report.payload, "agent_actions"
    )
    assert "apply_graphdb_import_after_approval" in string_list(
        report.payload, "agent_actions"
    )
    endpoint_action = action_plan["run_model_endpoint_smoke_after_credentials"]
    assert endpoint_action["status"] == "ready_after_user_action"
    assert endpoint_action["requires_user_action"] == "login_to_model_gateway_or_provide_token"
    assert string_list(endpoint_action, "command") == [
        "python3",
        "02.Source_code/asa_runtime/scripts/smoke_production_gateway_client.py",
        "--request",
        "/tmp/run/validated_request.json",
        "--output-dir",
        "/tmp/run/model_endpoint_smoke",
    ]
    chain_action = action_plan["run_remote_chain_after_approval"]
    assert chain_action["status"] == "ready_after_user_action"
    assert string_list(chain_action, "command") == [
        "python3",
        "02.Source_code/asa_runtime/scripts/run_remote_chain.py",
        "--manifest",
        "/tmp/run/remote_chain_manifest.json",
        "--out",
        "/tmp/run/remote_chain_result.json",
    ]
    graphdb_action = action_plan["apply_graphdb_import_after_approval"]
    assert graphdb_action["status"] == "blocked_on_missing_artifacts"
    assert string_list(graphdb_action, "missing_artifacts") == [
        "graphdb_import_bundle_dir"
    ]
    assert string_list(graphdb_action, "next_actions") == [
        "build_research_graphdb_import_bundle",
        "request_empty_neo4j_write_approval",
        "apply_graphdb_import_after_approval",
    ]


def test_production_readiness_builds_graphdb_apply_command_when_bundle_exists() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "graphdb_import_bundle_dir": "/tmp/run/research_graphdb",
    }

    report = assess_production_readiness_from_payloads(ledger)
    action_plan = actions_by_id(report.payload)

    graphdb_action = action_plan["apply_graphdb_import_after_approval"]
    assert graphdb_action["status"] == "ready_after_user_action"
    assert graphdb_action["requires_user_action"] == "approve_empty_neo4j_database_write"
    assert string_list(graphdb_action, "command") == [
        "python3",
        "02.Source_code/asa_runtime/scripts/apply_graphdb_import_bundle.py",
        "--bundle-dir",
        "/tmp/run/research_graphdb",
        "--database-name",
        "atomistic_sim_agent_knowledge",
        "--approve-write",
        "--out",
        "/tmp/run/graphdb_write_report.json",
    ]


def test_production_readiness_missing_remote_and_endpoint_actions_have_recovery_steps() -> None:
    report = assess_production_readiness_from_payloads(amorphous_blocked_ledger())
    action_plan = actions_by_id(report.payload)

    remote_action = action_plan["run_remote_chain_after_approval"]
    assert string_list(remote_action, "missing_artifacts") == [
        "remote_execution_manifest_path"
    ]
    assert string_list(remote_action, "next_actions") == [
        "prepare_agent_compute_bundle_with_ssh_target",
        "request_qa_review_before_remote_submission",
        "run_remote_chain_after_approval",
    ]

    endpoint_action = action_plan["run_model_endpoint_smoke_after_credentials"]
    assert string_list(endpoint_action, "missing_artifacts") == [
        "validated_request_path"
    ]
    assert string_list(endpoint_action, "next_actions") == [
        "persist_validated_request_from_agent_plan",
        "login_to_model_gateway_or_provide_token",
        "run_model_endpoint_smoke_after_credentials",
    ]
