from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sim_agent.agent_harness.tools import RuntimeToolCall, RuntimeToolResult, ToolDefinition, ToolRegistry, execute_runtime_tool
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .types import RuntimeTraceEvent


@dataclass(frozen=True, slots=True)
class ModelSelectedToolCall:
    tool_name: str
    arguments: JsonMap


@dataclass(frozen=True, slots=True)
class ModelToolChoiceBlocked(RuntimeError):
    blocker: str

    def __str__(self) -> str:
        return self.blocker


@dataclass(frozen=True, slots=True)
class AsaAgentSession:
    run_id: str
    session_id: str
    agent_id: str
    user_goal: str
    endpoint: ModelProviderConfig
    output_dir: Path
    registry: ToolRegistry

    def model_visible_tool_schemas(self) -> tuple[JsonMap, ...]:
        return tuple(_tool_schema(tool) for tool in self.registry.tools)

    def runtime_tool_call(self, selected: ModelSelectedToolCall) -> RuntimeToolCall:
        return RuntimeToolCall(
            tool_name=selected.tool_name,
            arguments=selected.arguments,
            run_id=self.run_id,
            session_id=self.session_id,
        )


class ToolChoiceModel(Protocol):
    model_id: str

    def choose_tools(
        self,
        session: AsaAgentSession,
        tool_schemas: Sequence[JsonMap],
    ) -> tuple[ModelSelectedToolCall, ...]:
        ...


@dataclass(frozen=True, slots=True)
class StaticToolChoiceModel:
    selected_tools: tuple[ModelSelectedToolCall, ...]
    model_id: str = "static-tool-choice-model"

    def choose_tools(
        self,
        _session: AsaAgentSession,
        _tool_schemas: Sequence[JsonMap],
    ) -> tuple[ModelSelectedToolCall, ...]:
        return self.selected_tools


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    session: AsaAgentSession
    model_id: str
    tool_schemas: tuple[JsonMap, ...]
    selected_tools: tuple[ModelSelectedToolCall, ...]
    tool_results: tuple[RuntimeToolResult, ...]
    trace: tuple[RuntimeTraceEvent, ...]
    status: str
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentLoop:
    session: AsaAgentSession
    model: ToolChoiceModel

    def run(self) -> AgentLoopResult:
        tool_schemas = self.session.model_visible_tool_schemas()
        model_id = _model_id(self.model, self.session)
        trace: list[RuntimeTraceEvent] = [
            RuntimeTraceEvent("asa_agent_session_created", self.session.agent_id, self.session.session_id),
            RuntimeTraceEvent("model_visible_tools_registered", self.session.agent_id, str(len(tool_schemas))),
        ]
        try:
            selected_tools = self.model.choose_tools(self.session, tool_schemas)
        except ModelToolChoiceBlocked as exc:
            blockers = (exc.blocker,)
            trace.append(RuntimeTraceEvent("agent_loop_model_selected_tools", self.session.agent_id, "0"))
            trace.append(RuntimeTraceEvent("model_tool_selection_blocked", self.session.agent_id, exc.blocker))
            trace.append(RuntimeTraceEvent("agent_loop_completed", self.session.agent_id, "blocked"))
            return AgentLoopResult(
                session=self.session,
                model_id=model_id,
                tool_schemas=tool_schemas,
                selected_tools=(),
                tool_results=(),
                trace=tuple(trace),
                status="blocked",
                blockers=blockers,
            )
        trace.append(RuntimeTraceEvent("agent_loop_model_selected_tools", self.session.agent_id, str(len(selected_tools))))
        tool_results: list[RuntimeToolResult] = []
        for selected in selected_tools:
            trace.append(RuntimeTraceEvent("model_tool_selected", self.session.agent_id, selected.tool_name))
            result = execute_runtime_tool(self.session.runtime_tool_call(selected), self.session.registry, self.session.output_dir)
            tool_results.append(result)
            trace.append(RuntimeTraceEvent("tool_executed", self.session.agent_id, f"{result.tool_name}:{result.status}"))
            if result.blocker:
                trace.append(RuntimeTraceEvent("tool_blocked", self.session.agent_id, f"{result.tool_name}:{result.blocker}"))
        blockers = tuple(result.blocker for result in tool_results if result.blocker is not None)
        if not selected_tools:
            blockers = ("no_model_tool_selected",)
        status = _loop_status(tuple(tool_results), blockers)
        trace.append(RuntimeTraceEvent("agent_loop_completed", self.session.agent_id, status))
        return AgentLoopResult(
            session=self.session,
            model_id=model_id,
            tool_schemas=tool_schemas,
            selected_tools=selected_tools,
            tool_results=tuple(tool_results),
            trace=tuple(trace),
            status=status,
            blockers=blockers,
        )


def _loop_status(tool_results: tuple[RuntimeToolResult, ...], blockers: tuple[str, ...]) -> str:
    if blockers:
        return "blocked"
    if all(result.status == "succeeded" for result in tool_results):
        return "succeeded"
    return "failed"


def _model_id(model: ToolChoiceModel, session: AsaAgentSession) -> str:
    model_id_for_session = getattr(model, "model_id_for_session", None)
    if callable(model_id_for_session):
        value = model_id_for_session(session)
        if isinstance(value, str) and value:
            return value
    return model.model_id


def _tool_schema(tool: ToolDefinition) -> JsonMap:
    return {
        "name": tool.name,
        "boundary": tool.boundary,
        "family": tool.family,
        "safety": tool.safety,
        "approval_required": tool.approval_required,
        "executable": tool.executable,
        "policy_id": tool.policy_id,
        "policy_summary": tool.policy_summary,
    }
