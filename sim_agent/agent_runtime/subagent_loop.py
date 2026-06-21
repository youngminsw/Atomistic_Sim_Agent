from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.agent_harness.tool_types import ToolRegistry
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .agent_registry import AgentSessionHandle
from .agent_specs import SubagentPresetSpec


PLACEHOLDER_GATEWAY_BASE_URL: Final = "https://model-gateway.local/v1"


@dataclass(frozen=True, slots=True)
class SubagentLoopRun:
    status: str
    model_id: str
    selected_tools: tuple[str, ...]
    blockers: tuple[str, ...]
    trace: tuple[JsonMap, ...]
    tool_result_refs: tuple[str, ...]


def run_subagent_agent_loop(
    handle: AgentSessionHandle,
    preset: SubagentPresetSpec,
    task_id: str,
    task: str,
    depth: int,
    subagent_dir: Path,
) -> SubagentLoopRun:
    from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession

    session = AsaAgentSession(
        run_id=f"subagent-{task_id}",
        session_id=f"{handle.agent_session_id}:subagent:{preset.name}:{task_id}",
        agent_id=f"{handle.agent_id}.{preset.name}.{task_id}",
        user_goal=_subagent_goal(handle, preset, task, depth),
        endpoint=_endpoint(handle),
        output_dir=subagent_dir,
        registry=_preset_tool_registry(preset),
    )
    result = AgentLoop(session, _subagent_model(handle, preset, task_id, task, session)).run()
    return SubagentLoopRun(
        status=result.status,
        model_id=result.model_id,
        selected_tools=tuple(call.tool_name for call in result.selected_tools),
        blockers=result.blockers,
        trace=tuple(
            {"event_type": event.event_type, "agent": event.agent, "summary": event.summary}
            for event in result.trace
        ),
        tool_result_refs=tuple(tool_result.artifact_ref for tool_result in result.tool_results),
    )


def _preset_tool_registry(preset: SubagentPresetSpec) -> ToolRegistry:
    from sim_agent.agent_harness.tools import default_tool_registry

    base = default_tool_registry()
    return ToolRegistry(tuple(base.require_tool(name) for name in preset.tool_names))


def _subagent_model(
    handle: AgentSessionHandle,
    preset: SubagentPresetSpec,
    task_id: str,
    task: str,
    session: AsaAgentSession,
):
    if _uses_static_fallback(handle):
        from sim_agent.agents_sdk_runtime import ModelSelectedToolCall, StaticToolChoiceModel

        return StaticToolChoiceModel(
            (
                ModelSelectedToolCall(
                    "artifact_write",
                    {
                        "relative_path": "subagent_report.md",
                        "content": _static_subagent_report(handle, preset, task_id, task),
                    },
                ),
            ),
            model_id=f"{handle.model.provider}/{handle.model.name}:subagent/{preset.name}",
        )
    from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel, provider_tool_choice_model_id

    return ProviderToolChoiceModel(model_id=f"{provider_tool_choice_model_id(session)}:subagent/{preset.name}")


def _subagent_goal(handle: AgentSessionHandle, preset: SubagentPresetSpec, task: str, depth: int) -> str:
    return "\n".join(
        (
            f"Caller agent: {handle.agent_id}",
            f"Bounded preset: {preset.name}",
            f"Role: {preset.role_prompt}",
            f"Scope: {preset.scope_notes}",
            f"Depth: {depth}",
            f"Task: {task}",
        )
    )


def _static_subagent_report(handle: AgentSessionHandle, preset: SubagentPresetSpec, task_id: str, task: str) -> str:
    return "\n".join(
        (
            f"# {preset.display_name} Bounded Run",
            "",
            f"caller_agent: {handle.agent_id}",
            f"preset: {preset.name}",
            f"task_id: {task_id}",
            "",
            task,
        )
    )


def _uses_static_fallback(handle: AgentSessionHandle) -> bool:
    return (
        handle.model.auth_mode == "none"
        or handle.model.provider in {"local_gateway", "offline", "static"}
        or handle.model.base_url == PLACEHOLDER_GATEWAY_BASE_URL
    )


def _endpoint(handle: AgentSessionHandle) -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": handle.model.provider,
            "model": handle.model.name,
            "reasoning_effort": handle.model.reasoning_effort,
            "base_url": handle.model.base_url,
            "auth_mode": handle.model.auth_mode,
            "api_key_env": handle.model.api_key_env,
        }
    )
