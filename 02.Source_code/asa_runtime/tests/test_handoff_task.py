from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agent_runtime import GlobalSessionModel, GlobalSessionOpenRequest, HandoffTaskRequest, handoff_task, open_global_session
from sim_agent.cli.tui_state import ModelSettings, TuiState, persist_state


def test_handoff_task_writes_target_session_and_handoff_ledger(tmp_path: Path) -> None:
    state = _static_state(tmp_path)

    result = handoff_task(
        state.session_dir,
        HandoffTaskRequest(
            from_agent="orchestrator",
            target_agent="md_agent",
            task_id="task-md-001",
            thread_id="thread-md",
            task="Plan a bounded MD campaign",
        ),
    )

    handoffs = _jsonl(state.session_dir / "message_bus" / "handoffs.jsonl")
    messages = _jsonl(state.session_dir / "agent_sessions" / "md_agent" / "messages.jsonl")
    events = _jsonl(state.session_dir / "agent_sessions" / "md_agent" / "events.jsonl")
    bus_records = _jsonl(state.session_dir / "message_bus" / "messages.jsonl")
    assert result.status == "succeeded"
    assert result.handoff_status == "live_completed"
    assert handoffs[-1]["task_id"] == "task-md-001"
    assert handoffs[-1]["status"] == "live_completed"
    assert handoffs[-1]["agent_loop_status"] == "succeeded"
    assert [record["record_type"] for record in bus_records] == ["send", "ack", "read", "reply"]
    assert any(message["role"] == "user" and message["content"] == "Plan a bounded MD campaign" for message in messages)
    assert messages[-1]["role"] == "assistant"
    assert "agent loop completed" in str(messages[-1]["content"])
    assert any(event["event_type"] == "agent_loop_completed" for event in events)
    assert events[-1]["event_type"] == "handoff_task_executed"


def test_handoff_task_blocks_unknown_target_with_durable_error(tmp_path: Path) -> None:
    state = _static_state(tmp_path)

    result = handoff_task(
        state.session_dir,
        HandoffTaskRequest(
            from_agent="orchestrator",
            target_agent="unknown_agent",
            task_id="task-bad",
            thread_id="thread-bad",
            task="Do impossible work",
        ),
    )

    errors = _jsonl(state.session_dir / "message_bus" / "handoff_errors.jsonl")
    assert result.status == "blocked"
    assert result.blocker == "unknown_agent"
    assert errors[-1]["task_id"] == "task-bad"
    assert errors[-1]["blocker"] == "unknown_agent"


def test_handoff_task_runtime_tool_executes_against_target_agent_session(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="handoff_task",
            arguments={
                "target_agent": "qa_agent",
                "task": "Audit the run evidence",
                "thread_id": "thread-qa",
                "task_id": "tool-task-qa",
            },
            run_id="handoff-run",
            session_id=state.session_id,
            caller_agent_id="orchestrator",
        ),
        registry,
        state.session_dir,
    )

    assert result.status == "succeeded"
    assert result.output["task_id"] == "tool-task-qa"
    assert result.output["target_agent"] == "qa_agent"
    assert result.output["status"] == "live_completed"
    assert "handoff_task" in registry.tool_names
    assert (state.session_dir / result.artifact_ref).is_file()
    messages = _jsonl(state.session_dir / "agent_sessions" / "qa_agent" / "messages.jsonl")
    assert any(message["role"] == "user" and message["content"] == "Audit the run evidence" for message in messages)
    assert messages[-1]["role"] == "assistant"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _static_state(session_dir: Path) -> TuiState:
    model = GlobalSessionModel(
        provider="static",
        name="explicit-static",
        reasoning_effort="high",
        base_url="https://model-gateway.local/v1",
        auth_mode="none",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )
    record = open_global_session(
        GlobalSessionOpenRequest(requested_dir=session_dir, default_root=session_dir, model=model)
    ).record
    state = TuiState(
        session_id=record.session_id,
        session_dir=record.session_dir,
        model=ModelSettings(
            provider=model.provider,
            name=model.name,
            reasoning_effort=model.reasoning_effort,
            base_url=model.base_url,
            auth_mode=model.auth_mode,
            api_key_env=model.api_key_env,
        ),
        global_session_id=record.session_id,
        global_session_path=record.paths.global_session,
    )
    persist_state(state)
    return state
