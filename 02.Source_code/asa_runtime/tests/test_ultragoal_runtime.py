from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agents_sdk_runtime import WorkflowHarnessResult


def test_ultragoal_materializes_durable_checkpoint_with_goal_context(tmp_path: Path) -> None:
    result = _ready_ultragoal(tmp_path)

    assert result.status == "ready"
    assert result.gate_status == "passed"
    assert result.artifact_refs == ("ultragoal/brief.md", "ultragoal/goals.json", "ultragoal/ledger.jsonl")
    goals = _read_json(tmp_path / "ultragoal" / "goals.json")
    assert goals["schema_version"] == "ultragoal_goals_v1"
    assert goals["active_goal_id"] == "goal-ultra-rich"
    assert goals["status"] == "active"
    assert goals["goals"][0]["status"] == "in_progress"
    checkpoint = _jsonl(tmp_path / "ultragoal" / "ledger.jsonl")[0]
    assert checkpoint["schema_version"] == "ultragoal_checkpoint_v1"
    assert checkpoint["checkpoint_id"] == "checkpoint-ultra-rich"
    assert checkpoint["active_goal_id"] == "goal-ultra-rich"
    assert checkpoint["signoff_gate_id"] == "signoff"
    assert isinstance(checkpoint["checkpoint_hash"], str) and len(checkpoint["checkpoint_hash"]) == 64


def test_ultragoal_checkpoint_is_idempotent_and_blocks_tampered_goals(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(gate=_signoff_gate())
    first = _ready_ultragoal(tmp_path)
    ledger = (tmp_path / "ultragoal" / "ledger.jsonl").read_text(encoding="utf-8")
    goals_path = tmp_path / "ultragoal" / "goals.json"
    goals_path.write_text('{"tampered": true}\n', encoding="utf-8")

    second = run_workflow_harness_smoke("ultragoal", payload, tmp_path)

    assert first.status == "ready"
    assert second.status == "blocked"
    assert second.gate_status == "blocked"
    assert second.blockers == ("ultragoal_artifact_conflict",)
    assert goals_path.read_text(encoding="utf-8") == '{"tampered": true}\n'
    assert (tmp_path / "ultragoal" / "ledger.jsonl").read_text(encoding="utf-8") == ledger


def test_ultragoal_signoff_gate_records_explicit_execution_decision(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "ultragoal",
        _payload(
            gate={
                "gate_id": "signoff",
                "gate_kind": "response_schema",
                "response_schema": {
                    "type": "object",
                    "required": ["decision"],
                    "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["approve", "decline"]},
                        "reason": {"type": "string"},
                    },
                },
            }
        ),
        tmp_path,
    )
    invalid = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ultragoal",
            "gate_id": "signoff",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "request-changes"},
        },
    )
    accepted = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ultragoal",
            "gate_id": "signoff",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "decline", "reason": "Need more evidence"},
        },
    )

    assert result.status == "blocked"
    assert result.gate_status == "awaiting_response"
    assert invalid.status == "blocked"
    assert invalid.blockers == ("ultragoal_signoff_response_mismatch",)
    assert accepted.status == "accepted"
    signoff = _read_json(tmp_path / "ultragoal" / "signoff.json")
    assert signoff["decision"] == "decline"
    assert signoff["approved"] is False
    assert signoff["reason"] == "Need more evidence"


def _payload(*, gate: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "ultra-rich",
        "user_goal": "Run verified workflow stories until signoff",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-ultra-rich",
        "evidence": {"codex_goal_snapshot": "active"},
        "goals": [{"id": "goal-ultra-rich", "title": "Close workflow parity gaps", "status": "in_progress"}],
    }
    if gate is not None:
        payload["gate"] = gate
    return payload


def _signoff_gate() -> dict[str, object]:
    return {
        "gate_id": "signoff",
        "gate_kind": "response_schema",
        "response_schema": {
            "type": "object",
            "required": ["decision"],
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "decline"]},
                "reason": {"type": "string"},
            },
        },
    }


def _ready_ultragoal(tmp_path: Path) -> WorkflowHarnessResult:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    payload = _payload(gate=_signoff_gate())
    pending = run_workflow_harness_smoke("ultragoal", payload, tmp_path)
    accepted = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ultragoal",
            "gate_id": "signoff",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "approve"},
            "idempotency_key": "ultragoal-test-approve",
        },
    )
    assert pending.status == "blocked"
    assert pending.gate_status == "awaiting_response"
    assert accepted.status == "accepted"
    return run_workflow_harness_smoke("ultragoal", payload, tmp_path)


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
