from __future__ import annotations

import json
import os
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agent_runtime import GlobalSessionModel, GlobalSessionOpenRequest, open_global_session
from sim_agent.cli.tui_state import ModelSettings, TuiState, persist_state


def test_synchronous_subagent_reports_terminal_lifecycle(tmp_path: Path) -> None:
    # Given: a bounded subagent task runs through the synchronous runtime.
    state = _static_state(tmp_path)

    # When: the caller creates and then lists the subagent job.
    run = _run_subagent(state, task_id="terminal-plan")
    listed = _control(state, "list")

    # Then: the ledger and list view both report terminal, non-controllable state.
    assert run.status == "succeeded"
    assert run.output["lifecycle"] == {
        "state": "completed",
        "running": False,
        "controllable": False,
    }
    assert listed.status == "succeeded"
    assert listed.output["subagents"][0]["state"] == "completed"
    assert listed.output["subagents"][0]["running"] is False
    assert listed.output["subagents"][0]["controllable"] is False


def test_completed_synchronous_subagent_is_not_controllable(tmp_path: Path) -> None:
    # Given: a synchronous bounded job has already reached terminal state.
    state = _static_state(tmp_path)
    _run_subagent(state, task_id="done-plan")
    subagent_dir = state.session_dir / "agent_sessions" / "md_agent" / "subagents" / "planner" / "done-plan"

    # When: the caller asks for a control operation that requires a live detached job.
    cancelled = _control(state, "cancel", subagent_id="done-plan")

    # Then: the API returns a typed blocker and does not append a fake control event.
    assert cancelled.status == "blocked"
    assert cancelled.blocker == "subagent_not_controllable"
    assert cancelled.output["state"] == "completed"
    assert cancelled.output["running"] is False
    assert cancelled.output["controllable"] is False
    assert not (subagent_dir / "subagent_controls.jsonl").exists()


def test_stale_running_lock_does_not_allow_control_success(tmp_path: Path) -> None:
    # Given: a lock-only job record exists but has no verified detached controller.
    state = _static_state(tmp_path)
    subagent_dir = state.session_dir / "agent_sessions" / "md_agent" / "subagents" / "planner" / "lock-only"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "subagent_running.lock").write_text(
        json.dumps(
            {
                "schema_version": "asa_subagent_running_lock_v1",
                "caller_agent": "md_agent",
                "preset": "planner",
                "subagent_id": "lock-only",
                "depth": 1,
                "task": "Pretend to be controllable.",
                "owner_pid": os.getpid(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the caller sends lifecycle controls to that lock-only record.
    paused = _control(state, "pause", subagent_id="lock-only")
    steered = _control(state, "steer", subagent_id="lock-only", content="Change direction.")
    progress = _control(state, "progress", subagent_id="lock-only")

    # Then: controls fail closed, while progress remains an honest snapshot.
    assert paused.status == "blocked"
    assert paused.blocker == "subagent_control_unsupported"
    assert paused.output["running"] is True
    assert paused.output["controllable"] is False
    assert paused.output["blocker"] == "subagent_control_unsupported"
    assert steered.status == "blocked"
    assert steered.blocker == "subagent_control_unsupported"
    assert progress.status == "succeeded"
    assert progress.output["state"] == "running"
    assert progress.output["running"] is True
    assert progress.output["controllable"] is False
    assert not (subagent_dir / "subagent_controls.jsonl").exists()


def test_corrupt_subagent_inspection_ledger_returns_typed_blocker(tmp_path: Path) -> None:
    # Given: a subagent ledger exists but contains malformed JSON.
    state = _static_state(tmp_path)
    subagent_dir = state.session_dir / "agent_sessions" / "md_agent" / "subagents" / "planner" / "corrupt-plan"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "subagent_run.json").write_text("{not-json", encoding="utf-8")

    # When: the caller inspects the job through the runtime tool.
    inspected = _inspect(state, subagent_id="corrupt-plan")

    # Then: the tool fails closed with a typed blocker instead of raising JSONDecodeError.
    assert inspected.status == "blocked"
    assert inspected.blocker == "corrupt_subagent_ledger"
    assert inspected.output["blocker"] == "corrupt_subagent_ledger"


def _run_subagent(state: TuiState, *, task_id: str):
    return execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "preset": "planner",
                "task_id": task_id,
                "task": "Plan a bounded MD window.",
            },
            run_id=f"subagent-{task_id}",
            session_id=state.session_id,
            caller_agent_id="md_agent",
        ),
        default_tool_registry(),
        state.session_dir,
    )


def _inspect(state: TuiState, *, subagent_id: str):
    return execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_inspect",
            arguments={"preset": "planner", "subagent_id": subagent_id},
            run_id=f"inspect-{subagent_id}",
            session_id=state.session_id,
            caller_agent_id="md_agent",
        ),
        default_tool_registry(),
        state.session_dir,
    )


def _control(state: TuiState, action: str, *, subagent_id: str = "", content: str = ""):
    arguments = {"action": action}
    if subagent_id:
        arguments.update({"preset": "planner", "subagent_id": subagent_id})
    if content:
        arguments["content"] = content
    return execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments=arguments,
            run_id=f"control-{action}-{subagent_id or 'all'}",
            session_id=state.session_id,
            caller_agent_id="md_agent",
        ),
        default_tool_registry(),
        state.session_dir,
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
