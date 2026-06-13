from __future__ import annotations

from sim_agent.llm_endpoints import AgentsSdkModelSpec

from .tools import ToolRegistry


class OfflineModelClient:
    def __init__(self) -> None:
        self._calls: list[str] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def plan(self, controller_name: str, model_spec: AgentsSdkModelSpec, registry: ToolRegistry) -> str:
        self._calls.append(f"plan:{controller_name}")
        return f"{model_spec.model}:{model_spec.reasoning_effort}:{len(registry.tools)}"
