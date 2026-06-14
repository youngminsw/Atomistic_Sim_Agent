from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    usage: str
    summary: str


COMMANDS: Final[tuple[SlashCommand, ...]] = (
    SlashCommand("/model", "/model status|set|login", "select gateway/model and manage OAuth/API credentials"),
    SlashCommand("/login", "/login [oauth|api-key] --provider <id>", "choose OAuth gateway or API key login flow"),
    SlashCommand("/hud", "/hud", "provider/model/auth/session HUD with connection guidance"),
    SlashCommand("/agents", "/agents", "show the orchestrator and specialist agent roster"),
    SlashCommand("/harness", "/harness", "show agent call matrix, QA gates, heartbeat, and recovery policy"),
    SlashCommand("/team", "/team [--output-dir PATH]", "start a team-session smoke and render agent activity"),
    SlashCommand("/team", "/team contract", "show heartbeat, timeout, call-matrix, and QA-gate contract"),
    SlashCommand("/skills", "/skills", "show simulation skills available to the agent team"),
    SlashCommand("/runtime", "/runtime [--output-dir PATH] [--smoke]", "exercise the OpenAI Agents SDK runtime path"),
    SlashCommand("/status", "/status", "show session, model, ledgers, and latest agent board"),
    SlashCommand("/log", "/log [--limit N]", "show recent session events"),
    SlashCommand("/run", "/run [--output-dir PATH] <goal>", "ask the main Orchestrator to prepare a run bundle"),
    SlashCommand("/ui", "/ui", "show the HTML controller launch command"),
    SlashCommand("/exit", "/exit", "close the interactive shell"),
)

SIMULATION_SKILLS: Final[tuple[tuple[str, str], ...]] = (
    ("research", "source-backed literature and GraphDB ingestion planning"),
    ("md", "LAMMPS structure, force-field, incident campaign, and physics gates"),
    ("ml-mdn", "MD event dataset, MDN training gate, uncertainty, active learning"),
    ("feature-scale", "KMC transport and Level-Set profile evolution"),
    ("qa", "hard-blocker audit, evidence gate, production-readiness report"),
    ("controller", "HTML controller bridge and run bundle inspection"),
)


def command_names() -> tuple[str, ...]:
    names = {command.name for command in COMMANDS}
    return tuple(sorted(names))


def suggested_commands(prefix: str) -> tuple[SlashCommand, ...]:
    if prefix == "/":
        return COMMANDS
    return tuple(command for command in COMMANDS if command.name.startswith(prefix) or prefix in command.usage)
