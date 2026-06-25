from __future__ import annotations

from sim_agent.production_readiness_ledger import (
    amorphous_prep_blockers,
    amorphous_prep_status,
    amorphous_prep_worker_present,
    mapping,
    positive_int,
    string_list,
    text,
)
from sim_agent.schemas._parse import JsonMap


OAUTH_LIKE_AUTH_MODES = frozenset({"oauth", "gateway"})


def assess_md(
    ledger: JsonMap,
    hard_blockers: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    md = mapping(ledger, "md")
    blockers = string_list(md, "hard_blockers")
    hard_blockers.extend(blockers)
    if md.get("production_ready") is True and not blockers:
        evidence.append("md_production_ready")
        return
    hard_blockers.append("md_production_not_ready")
    if "amorphous_lammps_structure_source_required" in blockers:
        _assess_amorphous_structure_prep(ledger, hard_blockers, agent_actions, evidence)
    agent_actions.append("resolve_md_production_blockers")


def _assess_amorphous_structure_prep(
    ledger: JsonMap,
    hard_blockers: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    prep_status = amorphous_prep_status(ledger)
    match prep_status:
        case "remote_plan_failed":
            evidence.append("amorphous_structure_prep_remote_failed")
            hard_blockers.extend(
                amorphous_prep_blockers(ledger)
                or ["amorphous_structure_prep_remote_failed"]
            )
            agent_actions.append("diagnose_amorphous_structure_prep_remote_failure")
        case "remote_plan_completed":
            evidence.append("amorphous_structure_prep_remote_completed")
            agent_actions.append("rerun_agent_with_relaxed_amorphous_structure_source")
        case "":
            assess_missing_amorphous_prep(ledger, agent_actions, evidence)
        case unreachable:
            hard_blockers.append(f"unknown_amorphous_structure_prep_status={unreachable}")
            agent_actions.append("diagnose_amorphous_structure_prep_remote_failure")


def assess_missing_amorphous_prep(
    ledger: JsonMap,
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    if amorphous_prep_worker_present(ledger):
        evidence.append("amorphous_structure_prep_worker_bundle_written")
        agent_actions.append("run_amorphous_structure_prep_worker_after_approval")
        agent_actions.append("rerun_agent_with_relaxed_amorphous_structure_source")
        return
    agent_actions.append("prepare_or_import_relaxed_amorphous_structure")


def assess_remote(
    ledger: JsonMap,
    hard_blockers: list[str],
    user_actions: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    remote = mapping(ledger, "remote")
    if text(remote, "chain_status") == "remote_chain_completed":
        evidence.append("remote_chain_completed")
        return
    hard_blockers.extend(string_list(remote, "chain_blockers") or ["remote_chain_not_completed"])
    user_actions.append("approve_remote_or_long_compute_run")
    agent_actions.append("run_remote_chain_after_approval")


def assess_surrogate(
    ledger: JsonMap,
    hard_blockers: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    surrogate = mapping(ledger, "surrogate")
    if surrogate.get("training_gate_accepted") is True:
        evidence.append("surrogate_training_gate_accepted")
        return
    hard_blockers.extend(
        string_list(surrogate, "training_gate_blockers")
        or ["surrogate_training_gate_not_accepted"]
    )
    agent_actions.append("train_or_active_learn_surrogate")


def assess_model_endpoint(
    ledger: JsonMap,
    endpoint_report: JsonMap,
    hard_blockers: list[str],
    user_actions: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    model_provider = mapping(ledger, "model_provider")
    auth_mode = text(model_provider, "auth_mode")
    if endpoint_smoke_passed(endpoint_report):
        evidence.append("model_endpoint_smoke_passed")
        return
    hard_blockers.append("model_endpoint_smoke_required")
    if auth_mode in OAUTH_LIKE_AUTH_MODES:
        user_actions.append("login_to_model_provider_or_provide_token")
    agent_actions.append("run_model_endpoint_smoke_after_credentials")


def endpoint_smoke_passed(endpoint_report: JsonMap) -> bool:
    direct_smoke_ok = endpoint_report.get("ok") is True and endpoint_report.get("real_endpoint_called") is True
    gateway_smoke_ok = (
        endpoint_report.get("ledger_version") == "production_gateway_smoke_v1"
        and endpoint_report.get("production_smoke") is True
        and endpoint_report.get("offline") is False
        and endpoint_report.get("fake_gateway_model") is False
        and endpoint_report.get("gateway_health_ok") is True
        and endpoint_report.get("endpoint_status") == 200
        and bool(text(endpoint_report, "gateway_request_id"))
        and not string_list(endpoint_report, "hard_blockers")
    )
    return direct_smoke_ok or gateway_smoke_ok


def assess_graphdb(
    graphdb_report: JsonMap,
    hard_blockers: list[str],
    user_actions: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    if graphdb_live_ingest_passed(graphdb_report):
        evidence.append("graphdb_live_ingest_passed")
        return
    hard_blockers.append("graphdb_live_ingest_required")
    user_actions.append("approve_empty_neo4j_database_write")
    agent_actions.append("apply_graphdb_import_after_approval")


def graphdb_live_ingest_passed(graphdb_report: JsonMap) -> bool:
    compact_report_ok = graphdb_report.get("accepted") is True and graphdb_report.get("live_write_completed") is True
    if compact_report_ok:
        return True
    row_counts = mapping(graphdb_report, "row_counts")
    required_imports = {"sources", "understandings", "claims", "entities"}
    executed_imports = set(string_list(graphdb_report, "executed_statement_kinds"))
    return (
        graphdb_report.get("applied") is True
        and graphdb_report.get("status") == "applied"
        and not string_list(graphdb_report, "blocker_reasons")
        and required_imports.issubset(executed_imports)
        and all(positive_int(row_counts, key) for key in required_imports)
    )


def assess_feature_scale(
    feature_report: JsonMap,
    hard_blockers: list[str],
    agent_actions: list[str],
    evidence: list[str],
) -> None:
    if feature_report.get("status") == "pass" and feature_report.get("evidence_scope") != "offline_demo_fixture":
        evidence.append("production_feature_scale_passed")
        return
    hard_blockers.append("production_feature_scale_report_required")
    agent_actions.append("run_feature_scale_from_accepted_production_surrogate")
