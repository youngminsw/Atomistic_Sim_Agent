from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.production_readiness_actions import (
    ACTION_BUILDERS,
    action_entry as _action_entry,
    action_plan,
)
from sim_agent.production_readiness_assessments import (
    assess_feature_scale,
    assess_graphdb,
    assess_md,
    assess_model_endpoint,
    assess_remote,
    assess_surrogate,
)
from sim_agent.production_readiness_contract import (
    PRODUCTION_ACTION_SPECS,
    UNKNOWN_ACTION_REPAIR_STEPS,
    action_plan_hard_blockers,
)
from sim_agent.production_readiness_ledger import (
    dedupe,
    read_json,
    read_optional_json,
    text,
)
from sim_agent.production_readiness_actions import _missing_action
from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class ProductionReadinessReport:
    production_ready: bool
    payload: JsonMap


def assess_production_readiness(
    ledger_path: Path,
    model_endpoint_smoke_report_path: Path | None = None,
    graphdb_ingest_report_path: Path | None = None,
    feature_qa_report_path: Path | None = None,
) -> ProductionReadinessReport:
    return assess_production_readiness_from_payloads(
        ledger=read_json(ledger_path, "agent_run_ledger"),
        model_endpoint_smoke_report=read_optional_json(model_endpoint_smoke_report_path),
        graphdb_ingest_report=read_optional_json(graphdb_ingest_report_path),
        feature_qa_report=read_optional_json(feature_qa_report_path),
    )


def assess_production_readiness_from_payloads(
    ledger: JsonMap,
    model_endpoint_smoke_report: JsonMap | None = None,
    graphdb_ingest_report: JsonMap | None = None,
    feature_qa_report: JsonMap | None = None,
) -> ProductionReadinessReport:
    hard_blockers: list[str] = []
    user_actions: list[str] = []
    agent_actions: list[str] = []
    evidence: list[str] = ["agent_run_ledger_loaded"]

    assess_md(ledger, hard_blockers, agent_actions, evidence)
    assess_remote(ledger, hard_blockers, user_actions, agent_actions, evidence)
    assess_surrogate(ledger, hard_blockers, agent_actions, evidence)
    assess_model_endpoint(
        ledger,
        model_endpoint_smoke_report or {},
        hard_blockers,
        user_actions,
        agent_actions,
        evidence,
    )
    assess_graphdb(
        graphdb_ingest_report or {},
        hard_blockers,
        user_actions,
        agent_actions,
        evidence,
    )
    assess_feature_scale(
        feature_qa_report or {},
        hard_blockers,
        agent_actions,
        evidence,
    )

    deduped_user_actions = dedupe(user_actions)
    deduped_agent_actions = dedupe(agent_actions)
    planned_actions = action_plan(ledger, deduped_agent_actions, deduped_user_actions)
    hard_blockers.extend(action_plan_hard_blockers(planned_actions))
    deduped_hard_blockers = dedupe(hard_blockers)
    ready = not deduped_hard_blockers and not deduped_user_actions
    return ProductionReadinessReport(
        production_ready=ready,
        payload={
            "report_version": "production_readiness_v1",
            "production_ready": ready,
            "run_id": text(ledger, "run_id"),
            "hard_blockers": deduped_hard_blockers,
            "user_actions": deduped_user_actions,
            "agent_actions": deduped_agent_actions,
            "action_plan": planned_actions,
            "evidence": dedupe(evidence),
        },
    )
