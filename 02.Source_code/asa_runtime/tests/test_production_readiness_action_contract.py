from __future__ import annotations

import sys
from pathlib import Path

from pytest import MonkeyPatch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import sim_agent.production_readiness as production_readiness
from production_readiness_fixtures import (
    JsonMap,
    actions_by_id,
    amorphous_blocked_ledger,
    string_list,
)
from sim_agent.production_readiness import (
    ACTION_BUILDERS,
    PRODUCTION_ACTION_SPECS,
    UNKNOWN_ACTION_REPAIR_STEPS,
    _action_entry,
    _missing_action,
    assess_production_readiness_from_payloads,
)


def test_production_readiness_assigns_owner_agents_to_actions() -> None:
    report = assess_production_readiness_from_payloads(amorphous_blocked_ledger())
    action_plan = actions_by_id(report.payload)

    assert action_plan["resolve_md_production_blockers"]["actor"] == "md_agent"
    assert action_plan["prepare_or_import_relaxed_amorphous_structure"]["actor"] == "md_agent"
    assert action_plan["run_remote_chain_after_approval"]["actor"] == "orchestrator"
    assert action_plan["train_or_active_learn_surrogate"]["actor"] == "ml_agent"
    assert action_plan["run_model_endpoint_smoke_after_credentials"]["actor"] == "orchestrator"
    assert action_plan["apply_graphdb_import_after_approval"]["actor"] == "research_agent"
    assert action_plan["run_feature_scale_from_accepted_production_surrogate"]["actor"] == "feature_scale_agent"


def test_production_readiness_action_specs_cover_emitted_actions() -> None:
    report = assess_production_readiness_from_payloads(amorphous_blocked_ledger())
    emitted_actions = string_list(report.payload, "agent_actions")

    assert set(ACTION_BUILDERS) == set(PRODUCTION_ACTION_SPECS)
    assert set(emitted_actions).issubset(PRODUCTION_ACTION_SPECS)
    for action_id, spec in PRODUCTION_ACTION_SPECS.items():
        assert spec.actor
        assert spec.missing_recovery_steps, action_id
        assert spec.missing_recovery_steps != UNKNOWN_ACTION_REPAIR_STEPS, action_id


def test_production_readiness_unknown_action_is_orchestrator_contract_error() -> None:
    action = _action_entry(amorphous_blocked_ledger(), "unexpected_new_action", [])

    assert action["actor"] == "orchestrator"
    assert action["status"] == "blocked_on_unknown_action_contract"
    assert action["hard_blockers"] == ["unknown_production_readiness_action=unexpected_new_action"]
    assert action["next_actions"] == [
        "repair_production_readiness_action_registry",
        "add_action_contract_regression_test",
    ]


def test_production_readiness_unknown_action_contract_promotes_top_level_blocker(
    monkeypatch: MonkeyPatch,
) -> None:
    def add_unknown_action(
        ledger: JsonMap,
        hard_blockers: list[str],
        agent_actions: list[str],
        evidence: list[str],
    ) -> None:
        agent_actions.append("unexpected_new_action")

    monkeypatch.setattr(production_readiness, "assess_feature_scale", add_unknown_action)

    report = production_readiness.assess_production_readiness_from_payloads(
        amorphous_blocked_ledger()
    )
    assert "unknown_production_readiness_action=unexpected_new_action" in string_list(
        report.payload, "hard_blockers"
    )
    assert report.payload["production_ready"] is False


def test_production_readiness_missing_action_always_has_recovery_steps() -> None:
    action = _missing_action("new_missing_contract_action", ["required_artifact"])

    assert action["status"] == "blocked_on_missing_artifacts"
    assert action["actor"] == "orchestrator"
    assert string_list(action, "next_actions") == [
        "repair_production_readiness_action_registry",
        "add_action_contract_regression_test",
    ]
