from __future__ import annotations

from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, RuntimeToolResult, ToolDefinition, ToolRegistry
from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession, ModelSelectedToolCall, StaticToolChoiceModel
from sim_agent.llm_endpoints import ModelProviderConfig


def test_blocked_tool_batch_does_not_execute_later_side_effects(tmp_path: Path) -> None:
    marker_path = tmp_path / "later-side-effect.txt"
    calls: list[str] = []
    session = _session(tmp_path, _registry(marker_path, calls))
    model = StaticToolChoiceModel(
        (
            ModelSelectedToolCall("blocking_tool", {}),
            ModelSelectedToolCall("side_effect_tool", {}),
        )
    )

    result = AgentLoop(session, model).run()

    assert result.status == "blocked"
    assert result.blockers == ("first_tool_blocked",)
    assert calls == ["blocking_tool"]
    assert tuple(tool_result.tool_name for tool_result in result.tool_results) == ("blocking_tool",)
    assert result.tool_results[0].blocker == "first_tool_blocked"
    assert not marker_path.exists()
    assert "first_tool_blocked" in {event.payload.get("summary") for event in result.runtime_events}


def test_unblocked_batch_preserves_ordered_results(tmp_path: Path) -> None:
    marker_path = tmp_path / "later-side-effect.txt"
    calls: list[str] = []
    session = _session(tmp_path, _registry(marker_path, calls, block_first=False))
    model = StaticToolChoiceModel(
        (
            ModelSelectedToolCall("blocking_tool", {}),
            ModelSelectedToolCall("side_effect_tool", {}),
        )
    )

    result = AgentLoop(session, model).run()

    assert result.status == "succeeded"
    assert result.blockers == ()
    assert calls == ["blocking_tool", "side_effect_tool"]
    assert [tool_result.tool_name for tool_result in result.tool_results] == [
        "blocking_tool",
        "side_effect_tool",
    ]
    assert [tool_result.output["order"] for tool_result in result.tool_results] == [1, 2]
    assert marker_path.read_text(encoding="utf-8") == "side effect"


def _session(tmp_path: Path, registry: ToolRegistry) -> AsaAgentSession:
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
        run_id="atomic-batch",
        session_id="asa-session-atomic-batch",
        agent_id="orchestrator",
        user_goal="Prove atomic tool batch handling",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=registry,
    )


def _registry(marker_path: Path, calls: list[str], *, block_first: bool = True) -> ToolRegistry:
    def first_tool(call: RuntimeToolCall, _session_dir: Path) -> RuntimeToolResult:
        calls.append(call.tool_name)
        blocker = "first_tool_blocked" if block_first else None
        return RuntimeToolResult(
            tool_name=call.tool_name,
            status="blocked" if blocker else "succeeded",
            output={"order": len(calls)},
            artifact_ref="",
            blocker=blocker,
        )

    def later_side_effect_tool(call: RuntimeToolCall, _session_dir: Path) -> RuntimeToolResult:
        calls.append(call.tool_name)
        marker_path.write_text("side effect", encoding="utf-8")
        return RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"order": len(calls)},
            artifact_ref=str(marker_path),
        )

    return ToolRegistry(
        tools=(
            ToolDefinition(
                "blocking_tool",
                "test",
                executable=True,
                approval_required=False,
                executor=first_tool,
            ),
            ToolDefinition(
                "side_effect_tool",
                "test",
                executable=True,
                approval_required=False,
                side_effect_class="test_marker_write",
                executor=later_side_effect_tool,
            ),
        )
    )
