from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.cli.tui_state import initial_state


@dataclass(frozen=True, slots=True)
class SubagentCallFixture:
    caller_agent: str
    preset: str
    task_id: str
    depth: int


def test_subagent_task_tool_runs_bounded_child_under_caller_session(tmp_path: Path) -> None:
    # Given: a persistent domain agent with the default runtime tools.
    state = initial_state(tmp_path)
    registry = default_tool_registry()

    # When: the agent invokes a bounded planner subagent.
    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "caller_agent": "md_agent",
                "preset": "planner",
                "task_id": "plan-md-window",
                "task": "Plan the MD event coverage window without mutating runtime state.",
            },
            run_id="subagent-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    # Then: the bounded run is durable under the caller/preset/id path.
    expected_dir = state.session_dir / "agent_sessions" / "md_agent" / "subagents" / "planner" / "plan-md-window"
    assert result.status == "succeeded"
    assert result.output["subagent_id"] == "plan-md-window"
    assert result.output["preset"] == "planner"
    assert result.output["caller_agent"] == "md_agent"
    assert result.output["depth"] == 1
    assert result.output["status"] == "succeeded"
    assert result.output["agent_loop_status"] == "succeeded"
    assert result.output["selected_tools"] == ["artifact_write"]
    assert result.output["session_dir"] == str(expected_dir)
    assert result.output["tool_names"] == ["artifact_write", "subagent_inspect"]
    assert (expected_dir / "subagent_run.json").is_file()
    assert (expected_dir / "messages.jsonl").is_file()
    assert (expected_dir / "artifacts" / "subagent_report.md").is_file()
    assert (state.session_dir / result.artifact_ref).is_file()
    run_payload = json.loads((expected_dir / "subagent_run.json").read_text(encoding="utf-8"))
    assert run_payload["agent_loop"]["selected_tools"] == ["artifact_write"]
    assert any(event["event_type"] == "asa_agent_session_created" for event in run_payload["agent_loop"]["trace"])


def test_subagent_task_tool_blocks_guardrail_violations(tmp_path: Path) -> None:
    # Given: one existing bounded child for a domain agent.
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    first = _subagent_call(state.session_id, SubagentCallFixture("md_agent", "planner", "duplicate-id", 1))
    assert execute_runtime_tool(first, registry, state.session_dir).status == "succeeded"

    # When: invalid bounded child requests are submitted.
    duplicate = execute_runtime_tool(first, registry, state.session_dir)
    unknown = execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("md_agent", "researcher", "unknown-preset", 1)),
        registry,
        state.session_dir,
    )
    too_deep = execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("md_agent", "planner", "too-deep", 2)),
        registry,
        state.session_dir,
    )
    self_call = execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("planner", "planner", "self-call", 1)),
        registry,
        state.session_dir,
    )

    # Then: each guardrail blocks with a durable reason.
    assert duplicate.blocker == "duplicate_task_id"
    assert unknown.blocker == "unknown_preset"
    assert too_deep.blocker == "subagent_depth_exceeded"
    assert self_call.blocker == "subagent_recursion_blocked"


def test_subagent_task_tool_allows_more_than_four_completed_children(tmp_path: Path) -> None:
    # Given: four completed bounded children for one caller.
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    for index in range(4):
        call = _subagent_call(state.session_id, SubagentCallFixture("qa_agent", "critic", f"completed-{index}", 1))
        result = execute_runtime_tool(call, registry, state.session_dir)
        assert result.status == "succeeded"

    # When: a fifth child is requested after the prior runs have completed.
    next_result = execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("qa_agent", "critic", "completed-4", 1)),
        registry,
        state.session_dir,
    )

    # Then: history remains inspectable but does not count as active work.
    assert next_result.status == "succeeded"


def test_subagent_task_tool_blocks_more_than_four_running_children(tmp_path: Path) -> None:
    # Given: four running bounded children for one caller.
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    for index in range(4):
        running_dir = state.session_dir / "agent_sessions" / "qa_agent" / "subagents" / "critic" / f"running-{index}"
        running_dir.mkdir(parents=True)
        (running_dir / "subagent_running.lock").write_text("{}\n", encoding="utf-8")

    # When: another child is requested while the running locks exist.
    overflow = execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("qa_agent", "critic", "running-4", 1)),
        registry,
        state.session_dir,
    )

    # Then: the per-caller active child cap blocks only concurrent child work.
    assert overflow.status == "blocked"
    assert overflow.blocker == "too_many_active_subagents"


def test_subagent_inspect_tool_reads_bounded_run_without_mutation(tmp_path: Path) -> None:
    # Given: a completed bounded executor run.
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    execute_runtime_tool(
        _subagent_call(state.session_id, SubagentCallFixture("feature_scale_agent", "executor", "execute-profile-plan", 1)),
        registry,
        state.session_dir,
    )

    # When: the read-only inspection tool reads that run.
    child_messages = (
        state.session_dir
        / "agent_sessions"
        / "feature_scale_agent"
        / "subagents"
        / "executor"
        / "execute-profile-plan"
        / "messages.jsonl"
    )
    before = child_messages.read_text(encoding="utf-8")
    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_inspect",
            arguments={
                "caller_agent": "feature_scale_agent",
                "preset": "executor",
                "subagent_id": "execute-profile-plan",
            },
            run_id="inspect-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )
    after = child_messages.read_text(encoding="utf-8")

    # Then: inspection returns parsed run data and does not append child messages.
    assert result.status == "succeeded"
    assert result.output["subagent_id"] == "execute-profile-plan"
    assert result.output["preset"] == "executor"
    assert before == after


def _subagent_call(
    session_id: str,
    fixture: SubagentCallFixture,
) -> RuntimeToolCall:
    return RuntimeToolCall(
        tool_name="subagent_task",
        arguments={
            "caller_agent": fixture.caller_agent,
            "preset": fixture.preset,
            "task_id": fixture.task_id,
            "task": f"Bounded task {fixture.task_id}",
            "depth": fixture.depth,
        },
        run_id=f"run-{fixture.task_id}",
        session_id=session_id,
    )
