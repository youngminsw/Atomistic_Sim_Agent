from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agent_runtime import GlobalSessionModel, GlobalSessionOpenRequest, open_global_session
from sim_agent.cli.tui_state import ModelSettings, TuiState, persist_state


def test_model_supplied_agent_identity_is_ignored(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()

    forged_message = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="agent_message",
            arguments={
                "action": "send",
                "from_agent": "qa_agent",
                "to_agent": "md_agent",
                "content": "forged sender",
                "thread_id": "forged-thread",
                "message_id": "forged-message",
            },
            run_id="forged-message-run",
            session_id=state.session_id,
            caller_agent_id="orchestrator",
        ),
        registry,
        state.session_dir,
    )
    forged_subagent = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "caller_agent": "qa_agent",
                "preset": "planner",
                "task_id": "forged-subagent",
                "task": "Attempt to create a child under another caller.",
            },
            run_id="forged-subagent-run",
            session_id=state.session_id,
            caller_agent_id="md_agent",
        ),
        registry,
        state.session_dir,
    )

    assert forged_message.status == "blocked"
    assert forged_message.blocker == "caller_identity_mismatch"
    assert forged_subagent.status == "blocked"
    assert forged_subagent.blocker == "caller_identity_mismatch"
    assert not (state.session_dir / "agent_sessions" / "qa_agent" / "subagents" / "planner" / "forged-subagent").exists()
    records_path = state.session_dir / "message_bus" / "messages.jsonl"
    if records_path.exists():
        assert all(record.get("message_id") != "forged-message" for record in _jsonl(records_path))


def test_missing_trusted_caller_identity_blocks_authority_tool(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="agent_message",
            arguments={
                "action": "send",
                "from_agent": "orchestrator",
                "to_agent": "md_agent",
                "content": "missing trusted caller",
                "message_id": "missing-trusted-caller",
            },
            run_id="missing-trusted-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert result.status == "blocked"
    assert result.blocker == "missing_trusted_caller_identity"
    records_path = state.session_dir / "message_bus" / "messages.jsonl"
    if records_path.exists():
        assert all(record.get("message_id") != "missing-trusted-caller" for record in _jsonl(records_path))


def test_trusted_caller_identity_sends_message(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="agent_message",
            arguments={
                "action": "send",
                "to_agent": "md_agent",
                "content": "trusted sender",
                "thread_id": "trusted-thread",
                "message_id": "trusted-message",
            },
            run_id="trusted-message-run",
            session_id=state.session_id,
            caller_agent_id="orchestrator",
        ),
        registry,
        state.session_dir,
    )

    records = _jsonl(state.session_dir / "message_bus" / "messages.jsonl")
    assert result.status == "succeeded"
    assert records[-1]["message_id"] == "trusted-message"
    assert records[-1]["from"] == "orchestrator"


def test_authority_tool_schemas_do_not_expose_spoofable_identity_fields() -> None:
    registry = default_tool_registry()
    schemas = {tool.name: tool.parameters for tool in registry.tools}

    for tool_name, fields in {
        "agent_message": {"from_agent", "by_agent"},
        "handoff_task": {"from_agent"},
        "subagent_task": {"caller_agent"},
        "subagent_inspect": {"caller_agent"},
        "subagent_control": {"caller_agent"},
    }.items():
        schema = schemas[tool_name]
        assert schema is not None
        properties = set(schema["properties"])
        assert properties.isdisjoint(fields)
        assert set(schema.get("required", ())).isdisjoint(fields)


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
