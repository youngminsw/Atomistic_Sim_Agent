from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class SubagentPresetSpec:
    name: str
    display_name: str
    role_prompt: str
    scope_notes: str
    tool_names: tuple[str, ...]
    persistent: bool
    clean_room: bool
    max_depth: int


SUBAGENT_PRESETS: Final = ("planner", "architect", "critic", "executor")
SUBAGENT_REPORT_TOOLS: Final = ("artifact_write", "subagent_inspect")

_PRESET_SPECS: Final = {
    "planner": SubagentPresetSpec(
        name="planner",
        display_name="Planner",
        role_prompt="Build bounded ASA execution plans with explicit assumptions, risks, and verification steps.",
        scope_notes="Planning, sequencing, acceptance criteria, and validation shape for ASA simulation work.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "architect": SubagentPresetSpec(
        name="architect",
        display_name="Architect",
        role_prompt="Analyze ASA runtime boundaries and propose minimal architecture-compatible changes.",
        scope_notes="Design boundaries, interfaces, contracts, and integration risk for ASA runtime modules.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "critic": SubagentPresetSpec(
        name="critic",
        display_name="Critic",
        role_prompt="Challenge ASA work products and identify concrete blockers before completion is claimed.",
        scope_notes="Review code, design, workflow, tool safety, evidence quality, and scientific validity.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "executor": SubagentPresetSpec(
        name="executor",
        display_name="Executor",
        role_prompt="Execute a bounded ASA task inside the caller's scoped run directory without shell access.",
        scope_notes="Implementation-shaped ASA work using model-visible tools, with no bash_process exposure.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
}


def resolve_subagent_preset(name: str) -> SubagentPresetSpec:
    try:
        return _PRESET_SPECS[name]
    except KeyError as exc:
        raise SchemaValidationError("unknown_preset") from exc
