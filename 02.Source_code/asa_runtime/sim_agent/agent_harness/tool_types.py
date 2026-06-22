from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


RuntimeToolExecutor = Callable[["RuntimeToolCall", Path], "RuntimeToolResult"]


@dataclass(frozen=True, slots=True)
class RuntimeToolError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class RuntimeToolCall:
    tool_name: str
    arguments: JsonMap
    run_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class RuntimeToolResult:
    tool_name: str
    status: str
    output: JsonMap
    artifact_ref: str
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    boundary: str
    family: str = "simulation"
    safety: str = "planning_only"
    approval_required: bool = True
    executable: bool = False
    policy_id: str = ""
    policy_summary: str = ""
    parameters: JsonMap | None = None
    strict: bool = True
    provenance: str = "asa_runtime"
    owner: str = "runtime"
    load_mode: str = "eager"
    side_effect_class: str = "none"
    concurrency_policy: str = "serial"
    result_serializer: str = "json"
    executor: RuntimeToolExecutor | None = None


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    tools: tuple[ToolDefinition, ...]

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.tools)

    def require_tool(self, name: str) -> ToolDefinition:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise RuntimeToolError("unknown_tool")
