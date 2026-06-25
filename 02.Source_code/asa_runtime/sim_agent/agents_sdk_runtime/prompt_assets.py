from __future__ import annotations

from importlib import resources
from typing import Final

PROMPT_RESOURCE_PACKAGE: Final = "sim_agent.agents_sdk_runtime"
PROMPT_ROOT: Final = "prompts"


def load_common_system_prompt() -> str:
    return _read_prompt("system/common_system.md")


def load_workflow_policy_prompt() -> str:
    return _read_prompt("system/workflow_policy.md")


def load_domain_role_prompt(agent_id: str) -> str:
    return _read_prompt(f"domain_roles/{agent_id}.md")


def load_subagent_role_prompt(preset_name: str) -> str:
    return _read_prompt(f"subagent_roles/{preset_name}.md")


def _read_prompt(relative_path: str) -> str:
    return (
        resources.files(PROMPT_RESOURCE_PACKAGE)
        .joinpath(PROMPT_ROOT, relative_path)
        .read_text(encoding="utf-8")
        .strip()
    )
