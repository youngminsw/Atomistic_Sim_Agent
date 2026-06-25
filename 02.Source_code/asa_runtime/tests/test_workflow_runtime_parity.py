from __future__ import annotations

import json
from pathlib import Path


def test_rich_gate_blocks_until_runtime_response_and_persists_flat_shape(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "rich-gate",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "orchestrator",
            "goal_id": "goal-rich-gate",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {
                "gate_id": "approval",
                "gate_kind": "enum",
                "allowed_values": ["approve", "revise"],
            },
        },
        tmp_path,
    )

    ledger = json.loads((tmp_path / result.ledger_ref).read_text(encoding="utf-8"))

    assert result.status == "blocked"
    assert result.gate_status == "awaiting_response"
    assert result.blockers == ("workflow_gate_response_required",)
    assert ledger["workflow_id"] == "ralplan"
    assert ledger["owner_agent_id"] == "orchestrator"
    assert ledger["target_agent_id"] == "orchestrator"
    assert ledger["goal_id"] == "goal-rich-gate"
    assert ledger["gate"]["schema_version"] == "workflow_gate_v1"
    assert ledger["gate"]["workflow_id"] == "ralplan"
    assert ledger["gate"]["goal_id"] == "goal-rich-gate"
    assert ledger["gate"]["gate_id"] == "approval"
    assert ledger["gate"]["gate_kind"] == "enum"
    assert ledger["gate"]["allowed_values"] == ["approve", "revise"]
    assert isinstance(ledger["gate"]["schema_hash"], str)


def test_workflow_gate_schema_hash_includes_goal_identity(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    first = run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "hash-a",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "orchestrator",
            "goal_id": "goal-a",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {"gate_id": "approval", "gate_kind": "enum", "allowed_values": ["approve", "revise"]},
        },
        tmp_path / "first",
    )
    second = run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "hash-b",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "orchestrator",
            "goal_id": "goal-b",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {"gate_id": "approval", "gate_kind": "enum", "allowed_values": ["approve", "revise"]},
        },
        tmp_path / "second",
    )
    first_ledger = json.loads((tmp_path / "first" / first.ledger_ref).read_text(encoding="utf-8"))
    second_ledger = json.loads((tmp_path / "second" / second.ledger_ref).read_text(encoding="utf-8"))

    assert first_ledger["gate"]["schema_hash"] != second_ledger["gate"]["schema_hash"]


def test_workflow_runtime_resumes_existing_gate_record(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    first = run_workflow_harness_smoke(
        "deep-interview",
        {
            "request_id": "resume-gate",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "research_agent",
            "goal_id": "goal-resume",
            "evidence": {"question_answer": "answer", "ambiguity_score": 0.1},
            "gate": {"gate_id": "clarify", "gate_kind": "enum", "allowed_values": ["clear", "blocked"]},
        },
        tmp_path,
    )
    second = run_workflow_harness_smoke(
        "deep-interview",
        {
            "request_id": "resume-gate",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "research_agent",
            "goal_id": "goal-resume",
            "evidence": {"question_answer": "answer", "ambiguity_score": 0.1},
            "gate": {"gate_id": "clarify", "gate_kind": "enum", "allowed_values": ["clear", "blocked"]},
        },
        tmp_path,
    )

    assert second.ledger_ref == first.ledger_ref
    assert second.status == "blocked"
    assert second.gate_status == "awaiting_response"


def test_ralplan_artifact_validation_blocks_missing_test_spec(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "prd.md").write_text("# PRD\n", encoding="utf-8")

    result = run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "ralplan-artifacts",
            "validate_artifact_paths": True,
            "artifact_root": str(artifact_root),
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
        },
        tmp_path,
    )

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ralplan_artifact_missing",)


def test_ultragoal_corrupt_goals_json_blocks_ready_state(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "goals.json").write_text("{not-json", encoding="utf-8")

    result = run_workflow_harness_smoke(
        "ultragoal",
        {
            "request_id": "ultragoal-corrupt",
            "artifact_root": str(artifact_root),
            "goals_path": "goals.json",
            "evidence": {"codex_goal_snapshot": {"status": "active"}},
        },
        tmp_path,
    )

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ultragoal_goals_corrupt",)


def test_ralplan_artifact_validation_blocks_missing_prd_or_test_spec(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "prd.md").write_text("# PRD\n", encoding="utf-8")

    result = run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "missing-artifact",
            "artifact_root": str(artifact_root),
            "validate_artifact_paths": True,
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
        },
        tmp_path / "workflows",
    )

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ralplan_artifact_missing",)
    assert result.missing_evidence == ("test_spec_path",)


def test_ultragoal_blocks_corrupt_goals_json_when_goals_path_is_supplied(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "goals.json").write_text("{not-json", encoding="utf-8")

    result = run_workflow_harness_smoke(
        "ultragoal",
        {
            "request_id": "corrupt-goals",
            "artifact_root": str(artifact_root),
            "goals_path": "goals.json",
            "evidence": {"codex_goal_snapshot": "present"},
        },
        tmp_path / "workflows",
    )

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ultragoal_goals_corrupt",)
