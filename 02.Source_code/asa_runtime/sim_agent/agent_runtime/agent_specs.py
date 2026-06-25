from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.agents_sdk_runtime.prompt_assets import load_subagent_role_prompt
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


SUBAGENT_PRESETS: Final = ("planner", "architect", "critic", "executor", "verifier")
SUBAGENT_REPORT_TOOLS: Final = ("artifact_write", "subagent_inspect")

_PRESET_SPECS: Final = {
    "planner": SubagentPresetSpec(
        name="planner",
        display_name="Planner",
        role_prompt=load_subagent_role_prompt("planner"),
        scope_notes="Planning, sequencing, acceptance criteria, and validation shape for ASA simulation work.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "architect": SubagentPresetSpec(
        name="architect",
        display_name="Architect",
        role_prompt=load_subagent_role_prompt("architect"),
        scope_notes="Design boundaries, interfaces, contracts, and integration risk for ASA runtime modules.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "critic": SubagentPresetSpec(
        name="critic",
        display_name="Critic",
        role_prompt=load_subagent_role_prompt("critic"),
        scope_notes="Review code, design, workflow, tool safety, evidence quality, and scientific validity.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "executor": SubagentPresetSpec(
        name="executor",
        display_name="Executor",
        role_prompt=load_subagent_role_prompt("executor"),
        scope_notes="Implementation-shaped ASA work using model-visible tools, with no bash_process exposure.",
        tool_names=SUBAGENT_REPORT_TOOLS,
        persistent=False,
        clean_room=True,
        max_depth=1,
    ),
    "verifier": SubagentPresetSpec(
        name="verifier",
        display_name="Verifier",
        role_prompt=load_subagent_role_prompt("verifier"),
        scope_notes="Completion evidence, ledger consistency, replay safety, and domain/runtime validity checks.",
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
