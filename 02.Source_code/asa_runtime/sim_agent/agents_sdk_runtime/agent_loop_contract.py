from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sim_agent.agent_harness.tools import RuntimeToolCall, RuntimeToolResult, ToolDefinition, ToolRegistry
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .types import RuntimeEvent, RuntimeTraceEvent


@dataclass(frozen=True, slots=True)
class ModelSelectedToolCall:
    tool_name: str
    arguments: JsonMap


@dataclass(frozen=True, slots=True)
class ModelTurnResult:
    selected_tools: tuple[ModelSelectedToolCall, ...] = ()
    final_output: str = ""


@dataclass(frozen=True, slots=True)
class ModelToolChoiceBlocked(RuntimeError):
    blocker: str

    def __str__(self) -> str:
        return self.blocker


@dataclass(slots=True)
class AsaAgentSession:
    run_id: str
    session_id: str
    agent_id: str
    user_goal: str
    endpoint: ModelProviderConfig
    output_dir: Path
    registry: ToolRegistry
    role_prompt: str = ""
    role_prompt_kind: str = "domain_role"
    workflow_policy: str = ""
    project_guidance: str = ""
    compact_summary: str = ""
    caller_context: str = ""
    raw_message_count: int = 0
    provider_context_blocker: str = ""
    compaction_metadata: JsonMap = field(default_factory=dict)
    workflow_state: JsonMap = field(default_factory=dict)
    skills: tuple[str, ...] = ()
    ledger_facts: list[JsonMap] = field(default_factory=list)
    messages: list[JsonMap] = field(default_factory=list)
    tool_history: list[JsonMap] = field(default_factory=list)

    def append_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def append_tool_result(self, result: RuntimeToolResult) -> None:
        self.tool_history.append(
            {
                "tool_name": result.tool_name,
                "status": result.status,
                "output": result.output,
                "artifact_ref": result.artifact_ref,
                "blocker": result.blocker,
            }
        )

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
    runtime_events: tuple[RuntimeEvent, ...] = ()
    final_output: str = ""


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
        "parameters": tool.parameters,
        "strict": tool.strict,
        "provenance": tool.provenance,
        "owner": tool.owner,
        "load_mode": tool.load_mode,
        "side_effect_class": tool.side_effect_class,
        "concurrency_policy": tool.concurrency_policy,
        "result_serializer": tool.result_serializer,
    }
