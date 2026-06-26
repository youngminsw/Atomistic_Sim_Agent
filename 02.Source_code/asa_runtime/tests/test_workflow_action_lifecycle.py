from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agents_sdk_runtime import (
    WorkflowHarnessResult,
    respond_workflow_gate,
    run_workflow_harness_smoke,
)
from sim_agent.cli.tui_state import initial_state


def test_pending_workflow_gate_creates_pending_action_file(tmp_path: Path) -> None:
    # Given: a workflow gate that blocks for a responder.
    result = _start_ralplan_gate(tmp_path)

    # When: the runtime persists the pending gate.
    action = _read_action(tmp_path, result.workflow_id, "approval")

    # Then: a sibling action record is available for lifecycle resolution.
    assert result.status == "blocked"
    assert action["schema_version"] == "workflow_action_v1"
    assert action["workflow_id"] == "ralplan"
    assert action["action_id"] == "approval"
    assert action["status"] == "pending"
    assert action["target_agent_id"] == "qa_agent"
    assert action["ledger_ref"] == "ralplan/actions/approval.json"


def test_resolver_unavailable_keeps_action_and_gate_pending(tmp_path: Path) -> None:
    # Given: a pending gate whose action resolver is unavailable.
    _start_ralplan_gate(tmp_path)
    _write_action_patch(tmp_path, "ralplan", "approval", {"resolver_available": False})

    # When: the target tries to resolve the gate.
    result = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "qa_agent",
            "value": "approve",
            "idempotency_key": "answer-1",
        },
    )

    # Then: the action remains pending and the gate is not falsely accepted.
    gate = _read_json(tmp_path / "ralplan" / "gates" / "approval.json")
    action = _read_action(tmp_path, "ralplan", "approval")
    assert result.status == "blocked"
    assert result.blockers == ("workflow_action_resolver_unavailable",)
    assert gate["status"] == "awaiting_response"
    assert action["status"] == "pending"
    assert result.to_json()["action_lifecycle"]["status"] == "blocked"


def test_idempotent_retry_same_key_and_value_is_duplicate_noop(tmp_path: Path) -> None:
    # Given: a pending gate resolved with an idempotency key.
    _start_ralplan_gate(tmp_path)
    first = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "qa_agent",
            "value": "approve",
            "idempotency_key": "answer-1",
        },
    )

    # When: the same responder retries with the same key and value.
    retry = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "qa_agent",
            "value": "approve",
            "idempotency_key": "answer-1",
        },
    )

    # Then: the retry is accepted as a duplicate no-op, not blocked as already answered.
    action = _read_action(tmp_path, "ralplan", "approval")
    assert first.status == "accepted"
    assert retry.status == "accepted"
    assert retry.blockers == ()
    assert retry.to_json()["action_lifecycle"]["status"] == "duplicate"
    assert action["status"] == "resolved"
    assert action["resolution"]["idempotency_key"] == "answer-1"


def test_idempotent_retry_same_key_different_value_is_conflict(tmp_path: Path) -> None:
    # Given: a pending gate resolved with an idempotency key.
    _start_ralplan_gate(tmp_path)
    respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "qa_agent",
            "value": "approve",
            "idempotency_key": "answer-1",
        },
    )

    # When: the same idempotency key is reused with a different value.
    conflict = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "qa_agent",
            "value": "revise",
            "idempotency_key": "answer-1",
        },
    )

    # Then: the retry is rejected as an idempotency conflict.
    action = _read_action(tmp_path, "ralplan", "approval")
    assert conflict.status == "blocked"
    assert conflict.blockers == ("workflow_action_idempotency_conflict",)
    assert conflict.to_json()["action_lifecycle"]["status"] == "blocked"
    assert action["resolution"]["value"] == "approve"


def test_deleted_action_state_before_reply_yields_workflow_action_unknown(tmp_path: Path) -> None:
    # Given: a pending gate whose action record was removed.
    _start_ralplan_gate(tmp_path)
    action_path = tmp_path / "ralplan" / "actions" / "approval.json"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text("{}\n", encoding="utf-8")
    action_path.unlink()

    # When: the target answers the still-pending gate.
    result = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "qa_agent", "value": "approve"},
    )

    # Then: the lifecycle broker rejects the unknown action.
    gate = _read_json(tmp_path / "ralplan" / "gates" / "approval.json")
    assert result.status == "blocked"
    assert result.blockers == ("workflow_action_unknown",)
    assert gate["status"] == "awaiting_response"
    assert result.to_json()["action_lifecycle"]["status"] == "blocked"


def test_corrupt_action_state_before_reply_yields_workflow_action_unknown(tmp_path: Path) -> None:
    # Given: a pending gate whose action record is corrupt.
    _start_ralplan_gate(tmp_path)
    action_path = tmp_path / "ralplan" / "actions" / "approval.json"
    action_path.write_text("{not-json\n", encoding="utf-8")

    # When: the target answers the still-pending gate.
    result = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "qa_agent", "value": "approve"},
    )

    # Then: the lifecycle broker rejects with a typed blocker instead of crashing.
    gate = _read_json(tmp_path / "ralplan" / "gates" / "approval.json")
    assert result.status == "blocked"
    assert result.blockers == ("workflow_action_unknown",)
    assert gate["status"] == "awaiting_response"
    assert result.to_json()["action_lifecycle"]["status"] == "blocked"


def test_workflow_gate_response_tool_accepts_idempotency_key(tmp_path: Path) -> None:
    # Given: a model-visible workflow gate response through the tool gateway.
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    execute_runtime_tool(
        RuntimeToolCall(
            tool_name="workflow_start",
            arguments={
                "workflow_id": "ralplan",
                "owner_agent_id": "orchestrator",
                "target_agent_id": "qa_agent",
                "goal_id": "goal-tool-action",
                "payload": {
                    "request_id": "workflow-action-tool",
                    "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
                    "gate": {"gate_id": "approval", "gate_kind": "enum", "allowed_values": ["approve", "revise"]},
                },
            },
            run_id="workflow-action-start",
            session_id=state.session_id,
            caller_agent_id="orchestrator",
        ),
        registry,
        state.session_dir,
    )

    # When: the model replies with an idempotency key.
    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="workflow_gate_response",
            arguments={
                "workflow_id": "ralplan",
                "gate_id": "approval",
                "value": "approve",
                "idempotency_key": "tool-answer-1",
            },
            run_id="workflow-action-response",
            session_id=state.session_id,
            caller_agent_id="qa_agent",
        ),
        registry,
        state.session_dir,
    )

    # Then: the key reaches the lifecycle broker and is reflected in tool JSON.
    assert result.status == "accepted"
    assert result.output["action_lifecycle"]["status"] == "resolved"
    assert result.output["action_lifecycle"]["idempotency_key"] == "tool-answer-1"


def _start_ralplan_gate(tmp_path: Path) -> WorkflowHarnessResult:
    return run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "action-lifecycle",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "qa_agent",
            "goal_id": "goal-action-lifecycle",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {"gate_id": "approval", "gate_kind": "enum", "allowed_values": ["approve", "revise"]},
        },
        tmp_path,
    )


def _read_action(tmp_path: Path, workflow_id: str, action_id: str) -> dict[str, object]:
    return _read_json(tmp_path / workflow_id / "actions" / f"{action_id}.json")


def _write_action_patch(tmp_path: Path, workflow_id: str, action_id: str, patch: dict[str, object]) -> None:
    path = tmp_path / workflow_id / "actions" / f"{action_id}.json"
    action = _read_json(path)
    path.write_text(json.dumps(action | patch, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
