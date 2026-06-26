from __future__ import annotations

import json
from pathlib import Path


def test_ralplan_materializes_consensus_stage_index_and_pending_approval(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("ralplan", _payload(), tmp_path)

    assert result.status == "ready"
    assert result.gate_status == "passed"
    assert "ralplan/consensus.json" in result.artifact_refs
    assert any(ref.endswith("/pending-approval.md") for ref in result.artifact_refs)
    consensus = _read_json(tmp_path / "ralplan" / "consensus.json")
    assert consensus["current_phase"] == "pending_approval"
    assert consensus["approved"] is False
    assert consensus["decision"] == "pending_approval"
    stages = consensus["stage_artifacts"]
    assert [stage["stage"] for stage in stages] == ["planner", "architect", "critic", "final"]
    assert all(isinstance(stage["sha256"], str) and len(stage["sha256"]) == 64 for stage in stages)
    index_lines = (tmp_path / "ralplan" / "plans" / "ralplan-rich" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 4
    assert json.loads(index_lines[-1])["stage"] == "final"
    assert (tmp_path / "ralplan" / "plans" / "ralplan-rich" / "pending-approval.md").is_file()


def test_ralplan_artifacts_are_idempotent_and_block_tampered_stage(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    first = run_workflow_harness_smoke("ralplan", _payload(), tmp_path)
    stage = tmp_path / "ralplan" / "plans" / "ralplan-rich" / "stage-01-planner.md"
    original_index = (tmp_path / "ralplan" / "plans" / "ralplan-rich" / "index.jsonl").read_text(encoding="utf-8")
    stage.write_text("# tampered\n", encoding="utf-8")

    second = run_workflow_harness_smoke("ralplan", _payload(), tmp_path)

    assert first.status == "ready"
    assert second.status == "blocked"
    assert second.gate_status == "blocked"
    assert second.blockers == ("ralplan_artifact_conflict",)
    assert stage.read_text(encoding="utf-8") == "# tampered\n"
    assert (tmp_path / "ralplan" / "plans" / "ralplan-rich" / "index.jsonl").read_text(encoding="utf-8") == original_index


def test_ralplan_approval_gate_requires_comments_for_request_changes_and_records_approval(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "ralplan",
        _payload(
            gate={
                "gate_id": "approval",
                "gate_kind": "response_schema",
                "response_schema": {
                    "type": "object",
                    "required": ["decision"],
                    "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["approve", "request-changes", "reject"]},
                        "comments": {"type": "string"},
                    },
                },
            }
        ),
        tmp_path,
    )

    missing_comments = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "request-changes"},
        },
    )
    accepted = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "approve"},
        },
    )

    assert result.status == "blocked"
    assert result.gate_status == "awaiting_response"
    assert missing_comments.status == "blocked"
    assert missing_comments.blockers == ("ralplan_approval_response_mismatch",)
    assert accepted.status == "accepted"
    approval = _read_json(tmp_path / "ralplan" / "approval.json")
    assert approval["decision"] == "approve"
    assert approval["approved"] is True
    assert approval["gate_id"] == "approval"


def _payload(*, gate: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "ralplan-rich",
        "user_goal": "Build a durable workflow runtime",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-ralplan-rich",
        "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
    }
    if gate is not None:
        payload["gate"] = gate
    return payload


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
