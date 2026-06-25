from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sim_agent.schemas._parse import JsonMap


PromptLayerKind = Literal[
    "system_policy",
    "workflow_policy",
    "domain_role",
    "subagent_role",
    "project_guidance",
    "compact_summary",
    "skills",
    "workflow_state",
    "ledger_facts",
    "tool_history",
]


@dataclass(frozen=True, slots=True)
class PromptLayer:
    kind: PromptLayerKind
    title: str
    content: str
    source: str

    def instruction_section(self) -> str:
        return f"[{self.kind}] {self.title}\n{self.content}"

    def to_json(self) -> JsonMap:
        return {
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "source": self.source,
        }


def prompt_layer(
    kind: PromptLayerKind,
    title: str,
    content: str,
    source: str,
) -> PromptLayer | None:
    cleaned = content.strip()
    if not cleaned:
        return None
    return PromptLayer(kind=kind, title=title, content=cleaned, source=source)
