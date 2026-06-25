from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agent_runtime import GlobalSessionModel, GlobalSessionOpenRequest, open_global_session
from sim_agent.cli.tui_state import ModelSettings, TuiState, persist_state


def test_subagent_control_lists_and_awaits_completed_bounded_job(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()
    run = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "caller_agent": "md_agent",
                "preset": "planner",
                "task_id": "plan-window",
                "task": "Plan a bounded MD window.",
            },
            run_id="subagent-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    listed = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={"action": "list", "caller_agent": "md_agent"},
            run_id="subagent-list",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )
    awaited = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "await",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "plan-window",
            },
            run_id="subagent-await",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert run.status == "succeeded"
    assert listed.status == "succeeded"
    assert listed.output["subagents"][0]["subagent_id"] == "plan-window"
    assert listed.output["subagents"][0]["state"] == "completed"
    assert awaited.status == "succeeded"
    assert awaited.output["state"] == "completed"
    assert awaited.output["running"] is False


def test_subagent_control_writes_lifecycle_events_for_running_job(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()
    running_dir = state.session_dir / "agent_sessions" / "qa_agent" / "subagents" / "critic" / "review-live"
    running_dir.mkdir(parents=True)
    (running_dir / "subagent_running.lock").write_text(
        json.dumps(
            {
                "caller_agent": "qa_agent",
                "preset": "critic",
                "subagent_id": "review-live",
                "depth": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paused = _control(state.session_id, state.session_dir, "pause", content="", registry=registry)
    steered = _control(state.session_id, state.session_dir, "steer", content="Check code and evidence together.", registry=registry)
    cancelled = _control(state.session_id, state.session_dir, "cancel", content="", registry=registry)
    progress = _control(state.session_id, state.session_dir, "progress", content="", registry=registry)

    assert paused.status == "succeeded"
    assert paused.output["state"] == "paused"
    assert steered.status == "succeeded"
    assert cancelled.status == "succeeded"
    assert cancelled.output["state"] == "cancel_requested"
    assert progress.status == "succeeded"
    assert progress.output["state"] == "cancel_requested"
    controls = (running_dir / "subagent_controls.jsonl").read_text(encoding="utf-8")
    assert "pause" in controls
    assert "steer" in controls
    assert "Check code and evidence together." in controls
    assert "cancel" in controls


def test_subagent_control_blocks_terminal_and_unknown_jobs(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()
    execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "caller_agent": "md_agent",
                "preset": "planner",
                "task_id": "done-job",
                "task": "Complete a bounded job.",
            },
            run_id="subagent-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    terminal_cancel = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "cancel",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "done-job",
            },
            run_id="terminal-cancel",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )
    missing = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "progress",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "missing-job",
            },
            run_id="missing-progress",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert terminal_cancel.status == "blocked"
    assert terminal_cancel.blocker == "subagent_already_terminal"
    assert missing.status == "blocked"
    assert missing.blocker == "unknown_subagent"


def test_subagent_control_detects_lost_process_and_restarts_same_job(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    registry = default_tool_registry()
    running_dir = state.session_dir / "agent_sessions" / "md_agent" / "subagents" / "planner" / "lost-plan"
    running_dir.mkdir(parents=True)
    (running_dir / "subagent_running.lock").write_text(
        json.dumps(
            {
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "lost-plan",
                "depth": 1,
                "task": "Recover a lost bounded planner run.",
                "owner_pid": 999999999,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    listed = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={"action": "list", "caller_agent": "md_agent"},
            run_id="lost-list",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )
    progress = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "progress",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "lost-plan",
            },
            run_id="lost-progress",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )
    restarted = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "restart",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "lost-plan",
            },
            run_id="lost-restart",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert listed.status == "succeeded"
    assert listed.output["subagents"][0]["state"] == "lost_process"
    assert listed.output["subagents"][0]["lost_process"] is True
    assert progress.status == "blocked"
    assert progress.blocker == "subagent_lost_process"
    assert progress.output["state"] == "lost_process"
    assert restarted.status == "succeeded"
    assert restarted.output["action"] == "restart"
    assert restarted.output["subagent_id"] == "lost-plan"
    assert restarted.output["previous_blocker"] == "subagent_lost_process"
    assert (running_dir / "subagent_run.json").is_file()
    assert any(path.name.startswith("lost-plan.lost-") for path in running_dir.parent.iterdir())


def test_provider_visible_schema_includes_subagent_control(tmp_path: Path) -> None:
    state = _static_state(tmp_path)
    schema = next(tool for tool in default_tool_registry().tools if tool.name == "subagent_control")

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={"action": "list", "caller_agent": "orchestrator"},
            run_id="schema-smoke",
            session_id=state.session_id,
        ),
        default_tool_registry(),
        state.session_dir,
    )

    assert schema.executable is True
    assert schema.approval_required is False
    assert result.status == "succeeded"


def _control(session_id: str, session_dir: Path, action: str, *, content: str, registry):
    return execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": action,
                "caller_agent": "qa_agent",
                "preset": "critic",
                "subagent_id": "review-live",
                "content": content,
            },
            run_id=f"control-{action}",
            session_id=session_id,
        ),
        registry,
        session_dir,
    )


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
