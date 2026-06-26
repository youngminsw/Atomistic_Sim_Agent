from __future__ import annotations

import json
from pathlib import Path


def test_workflow_goal_authority_allows_domain_self_and_orchestrator_child(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import adjust_workflow_goal_state

    domain_self = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "md_agent",
            "workflow_id": "ultrawork",
            "goal_id": "goal-md",
            "state": "active",
        },
    )
    orchestrator_child = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultraqa",
            "goal_id": "goal-qa",
            "state": "active",
        },
    )

    assert domain_self.status == "accepted"
    assert domain_self.blockers == ()
    assert orchestrator_child.status == "accepted"
    assert orchestrator_child.blockers == ()


def test_workflow_goal_authority_denies_peer_and_domain_to_orchestrator(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import adjust_workflow_goal_state

    peer = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultrawork",
            "goal_id": "goal-peer",
            "state": "active",
        },
    )
    to_orchestrator = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "qa_agent",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "orchestrator",
            "workflow_id": "ultraqa",
            "goal_id": "goal-orchestrator",
            "state": "active",
        },
    )

    assert peer.status == "blocked"
    assert peer.blockers == ("workflow_authority_peer_denied",)
    assert to_orchestrator.status == "blocked"
    assert to_orchestrator.blockers == ("workflow_authority_orchestrator_denied",)


def test_workflow_goal_authority_uses_stored_owner_for_existing_goal(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import adjust_workflow_goal_state, operate_workflow_goal

    create = operate_workflow_goal(
        tmp_path,
        {
            "operation": "create",
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "md_agent",
            "workflow_id": "ultrawork",
            "goal_id": "goal-owned",
            "objective": "Self-owned MD goal",
        },
    )
    peer = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "qa_agent",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultrawork",
            "goal_id": "goal-owned",
            "state": "complete",
        },
    )
    goal = json.loads((tmp_path / "ultrawork" / "goals" / "goal-owned.json").read_text(encoding="utf-8"))

    assert create.status == "accepted"
    assert peer.status == "blocked"
    assert peer.blockers == ("workflow_authority_peer_denied",)
    assert goal["owner_agent_id"] == "md_agent"
    assert goal["target_agent_id"] == "md_agent"
    assert goal["state"] == "active"
    assert [event["operation"] for event in goal["history"]] == ["create"]


def test_workflow_goal_authority_denies_orchestrator_cross_domain_mismatch(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import adjust_workflow_goal_state, run_workflow_harness_smoke

    direct = adjust_workflow_goal_state(
        tmp_path,
        {
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "md_agent",
            "workflow_id": "ultrawork",
            "goal_id": "goal-cross-domain",
            "state": "active",
        },
    )
    harness = run_workflow_harness_smoke(
        "ultrawork",
        {
            "request_id": "cross-domain-start",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "md_agent",
            "goal_id": "goal-cross-domain-start",
            "evidence": {"lane_outputs": ["lane-a"]},
        },
        tmp_path / "harness",
    )

    assert direct.status == "blocked"
    assert direct.blockers == ("workflow_authority_peer_denied",)
    assert harness.status == "blocked"
    assert harness.blockers == ("workflow_authority_peer_denied",)


def test_workflow_start_enforces_owner_scoped_authority(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    peer = run_workflow_harness_smoke(
        "ultrawork",
        {
            "request_id": "peer-start",
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "qa_agent",
            "goal_id": "goal-peer-start",
            "evidence": {"lane_outputs": ["lane-a"]},
        },
        tmp_path / "peer",
    )
    to_orchestrator = run_workflow_harness_smoke(
        "ultraqa",
        {
            "request_id": "orchestrator-start",
            "actor_agent_id": "qa_agent",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "orchestrator",
            "goal_id": "goal-orchestrator-start",
            "evidence": {"adversarial_scenarios": ["case-a"]},
        },
        tmp_path / "orchestrator",
    )

    assert peer.status == "blocked"
    assert peer.blockers == ("workflow_authority_peer_denied",)
    assert to_orchestrator.status == "blocked"
    assert to_orchestrator.blockers == ("workflow_authority_orchestrator_denied",)


def test_workflow_start_authority_denial_precedes_missing_evidence(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "ultrawork",
        {
            "request_id": "peer-start-no-evidence",
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "qa_agent",
            "goal_id": "goal-peer-start-no-evidence",
        },
        tmp_path,
    )

    assert result.status == "blocked"
    assert result.blockers == ("workflow_authority_peer_denied",)
    assert result.missing_evidence == ()
