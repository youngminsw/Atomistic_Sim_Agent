from __future__ import annotations

from pathlib import Path

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession, ModelSelectedToolCall, StaticToolChoiceModel
from sim_agent.agents_sdk_runtime.runtime_events import project_runtime_events, validate_runtime_event_order
from sim_agent.agents_sdk_runtime.types import RuntimeEventType
from sim_agent.llm_endpoints import ModelProviderConfig


def test_agent_loop_exposes_model_visible_tools_and_executes_selected_calls(tmp_path: Path) -> None:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "local_gateway",
            "model": "gpt-5.3",
            "reasoning_effort": "high",
            "base_url": "http://local-gateway.test/v1",
            "auth_mode": "none",
        }
    )
    session = AsaAgentSession(
        run_id="loop-smoke",
        session_id="asa-session-loop-smoke",
        agent_id="orchestrator",
        user_goal="Prove the model-selected tool loop",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )
    model = StaticToolChoiceModel(
        (
            ModelSelectedToolCall("artifact_write", {"relative_path": "loop/evidence.txt", "content": "loop-ok"}),
            ModelSelectedToolCall("graphdb_dry_run", {"database_name": "atomistic_sim_agent_knowledge"}),
        ),
        model_id="local_gateway/gpt-5.3",
    )

    result = AgentLoop(session, model).run()
    schema_by_name = {schema["name"]: schema for schema in result.tool_schemas}

    assert result.status == "succeeded"
    assert result.model_id == "local_gateway/gpt-5.3"
    assert tuple(call.tool_name for call in result.selected_tools) == ("artifact_write", "graphdb_dry_run")
    assert schema_by_name["artifact_write"]["executable"] is True
    assert schema_by_name["validate_simulation_request"]["executable"] is False
    assert [tool_result.status for tool_result in result.tool_results] == ["succeeded", "succeeded"]
    assert (tmp_path / "artifacts" / "loop" / "evidence.txt").read_text(encoding="utf-8") == "loop-ok"
    assert {event.event_type for event in result.trace} >= {
        "asa_agent_session_created",
        "model_visible_tools_registered",
        "model_tool_selected",
        "tool_executed",
        "agent_loop_completed",
    }


def test_agent_loop_blocks_when_model_selects_no_tools(tmp_path: Path) -> None:
    session = _session(tmp_path)

    result = AgentLoop(session, StaticToolChoiceModel(())).run()

    assert result.status == "blocked"
    assert result.blockers == ("no_model_tool_selected",)
    assert result.tool_results == ()
    assert validate_runtime_event_order(result.runtime_events) == result.runtime_events
    assert result.runtime_events[-1].event_type == RuntimeEventType.BLOCKER
    assert result.runtime_events[-1].payload["summary"] == "no_model_tool_selected"


def test_agent_loop_can_continue_after_tool_result_before_final_output(tmp_path: Path) -> None:
    session = _session(tmp_path)
    model = _ContinuingToolChoiceModel()

    result = AgentLoop(session, model).run()
    event_types = [event.event_type for event in validate_runtime_event_order(result.runtime_events)]
    projections = project_runtime_events(result.runtime_events)

    assert result.status == "succeeded"
    assert model.calls == 2
    assert tuple(call.tool_name for call in result.selected_tools) == ("artifact_write",)
    assert result.final_output == "tool result observed"
    assert session.tool_history[0]["tool_name"] == "artifact_write"
    assert session.messages[-1] == {"role": "assistant", "content": "tool result observed"}
    assert [event.event_type for event in result.trace].count("agent_loop_model_step_started") == 2
    assert "tool_result_appended" in {event.event_type for event in result.trace}
    assert event_types.count(RuntimeEventType.MODEL_START) == 2
    assert event_types.count(RuntimeEventType.MODEL_END) == 2
    assert RuntimeEventType.TOOL_START in event_types
    assert RuntimeEventType.TOOL_END in event_types
    assert RuntimeEventType.MESSAGE_APPEND in event_types
    assert any(projection.label == "artifact_write tool completed" for projection in projections)
    assert result.runtime_events[-1].event_type == RuntimeEventType.MESSAGE_APPEND


def test_agent_loop_can_cancel_before_model_or_tool_execution(tmp_path: Path) -> None:
    session = _session(tmp_path)
    model = StaticToolChoiceModel(
        (ModelSelectedToolCall("artifact_write", {"relative_path": "loop/cancelled.txt", "content": "no"}),)
    )

    result = AgentLoop(session, model, cancel_requested=True).run()

    assert result.status == "cancelled"
    assert result.blockers == ("cancelled",)
    assert result.selected_tools == ()
    assert result.tool_results == ()
    assert not (tmp_path / "artifacts" / "loop" / "cancelled.txt").exists()
    assert [event.event_type for event in result.runtime_events] == [RuntimeEventType.CANCELLATION]


def test_agent_loop_blocks_bounded_continuation_overflow_with_runtime_event(tmp_path: Path) -> None:
    session = _session(tmp_path)
    model = _NeverEndingContinuationModel()

    result = AgentLoop(session, model, max_steps=1).run()
    event_types = [event.event_type for event in validate_runtime_event_order(result.runtime_events)]

    assert result.status == "blocked"
    assert result.blockers == ("max_agent_loop_steps_exceeded",)
    assert model.calls == 1
    assert tuple(call.tool_name for call in result.selected_tools) == ("artifact_write",)
    assert event_types.count(RuntimeEventType.MODEL_START) == 1
    assert event_types.count(RuntimeEventType.TOOL_START) == 1
    assert event_types.count(RuntimeEventType.TOOL_END) == 1
    assert result.runtime_events[-1].event_type == RuntimeEventType.BLOCKER
    assert result.runtime_events[-1].payload["summary"] == "max_agent_loop_steps_exceeded"


def test_agent_loop_records_tool_execution_blocker_as_runtime_event(tmp_path: Path) -> None:
    session = _session(tmp_path)
    model = StaticToolChoiceModel((ModelSelectedToolCall("artifact_write", {"content": "missing path"}),))

    result = AgentLoop(session, model).run()
    event_types = [event.event_type for event in validate_runtime_event_order(result.runtime_events)]

    assert result.status == "blocked"
    assert result.tool_results[0].blocker is not None
    assert RuntimeEventType.TOOL_START in event_types
    assert RuntimeEventType.TOOL_END in event_types
    assert result.runtime_events[-1].event_type == RuntimeEventType.BLOCKER
    assert result.runtime_events[-1].payload["tool_name"] == "artifact_write"


def _session(tmp_path: Path) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "local_gateway",
            "model": "gpt-5.3",
            "reasoning_effort": "high",
            "base_url": "http://local-gateway.test/v1",
            "auth_mode": "none",
        }
    )
    return AsaAgentSession(
        run_id="loop-smoke",
        session_id="asa-session-loop-smoke",
        agent_id="orchestrator",
        user_goal="Prove the model-selected tool loop",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )


class _ContinuingToolChoiceModel:
    model_id = "continuing-tool-choice-model"
    supports_tool_result_continuation = True

    def __init__(self) -> None:
        self.calls = 0

    def choose_tools(
        self,
        session: AsaAgentSession,
        _tool_schemas: object,
    ) -> tuple[ModelSelectedToolCall, ...]:
        self.calls += 1
        if self.calls == 1:
            return (
                ModelSelectedToolCall(
                    "artifact_write",
                    {"relative_path": "loop/continued.txt", "content": "continued"},
                ),
            )
        assert session.tool_history[0]["tool_name"] == "artifact_write"
        return ()

    def final_output_for_session(self, _session: AsaAgentSession, _tool_results: object) -> str:
        return "tool result observed"


class _NeverEndingContinuationModel:
    model_id = "never-ending-tool-choice-model"
    supports_tool_result_continuation = True

    def __init__(self) -> None:
        self.calls = 0

    def choose_tools(
        self,
        _session: AsaAgentSession,
        _tool_schemas: object,
    ) -> tuple[ModelSelectedToolCall, ...]:
        self.calls += 1
        return (
            ModelSelectedToolCall(
                "artifact_write",
                {"relative_path": f"loop/overflow-{self.calls}.txt", "content": "overflow"},
            ),
        )
