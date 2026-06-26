from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    usage: str
    summary: str


COMMANDS: Final[tuple[SlashCommand, ...]] = (
    SlashCommand("/chat", "/chat [@agent] <message>|clear", "chat with Orchestrator, summon agents, and show transcript"),
    SlashCommand(
        "/model",
        "/model status|set|profiles|profile|assign|login",
        "select profiles, gateway/model, thinking, agent models, and credentials",
    ),
    SlashCommand("/login", "/login [oauth|api-key] --provider <id>", "choose browser OAuth or API key provider login"),
    SlashCommand("/hud", "/hud", "provider/model/auth/session HUD with connection guidance"),
    SlashCommand("/guide", "/guide", "plain-language Korean onboarding and next-step guide"),
    SlashCommand("/start", "/start", "same beginner onboarding as /guide"),
    SlashCommand("/wizard", "/wizard", "arrow-key setup for endpoint, login, GraphDB, memory seed, and interview-run"),
    SlashCommand("/agents", "/agents", "show the orchestrator and specialist agent roster"),
    SlashCommand("/compact", "/compact [status|replay] [@agent|agent]", "manual agent-session compaction and replay gate"),
    SlashCommand("/harness", "/harness", "show agent call matrix, QA gates, heartbeat, and recovery policy"),
    SlashCommand(
        "/workflow",
        "/workflow <name> [--gate-id ID] [--owner-agent AGENT] [--target-agent AGENT] [--output-dir PATH]",
        "start a resumable workflow harness for deep-interview, ralplan, ultragoal, visual-qa, or ultraresearch",
    ),
    SlashCommand(
        "/workflow-response",
        "/workflow-response <gate-id> <value> [--workflow-id NAME] [--responder-agent AGENT]",
        "answer a pending workflow gate and update the workflow gate ledger",
    ),
    SlashCommand("/deep-interview", "/deep-interview [--output-dir PATH]", "shortcut for the ambiguity-gated interview harness"),
    SlashCommand("/ralplan", "/ralplan [--output-dir PATH]", "shortcut for the RALPlan workflow harness"),
    SlashCommand("/ultrawork", "/ultrawork [--output-dir PATH]", "shortcut for parallel work orchestration harness"),
    SlashCommand("/ultraqa", "/ultraqa [--output-dir PATH]", "shortcut for adversarial QA workflow harness"),
    SlashCommand("/ultragoal", "/ultragoal [--output-dir PATH]", "shortcut for durable goal checkpoint harness"),
    SlashCommand("/visual-qa", "/visual-qa [--output-dir PATH]", "shortcut for evidence-captured visual QA workflow"),
    SlashCommand(
        "/ultraresearch",
        "/ultraresearch [--output-dir PATH]",
        "shortcut for public-source ultraresearch workflow with insane_search acquisition",
    ),
    SlashCommand("/team", "/team [--output-dir PATH]", "start a team-session smoke and render agent activity"),
    SlashCommand("/team", "/team contract", "show heartbeat, timeout, call-matrix, and QA-gate contract"),
    SlashCommand("/skills", "/skills", "show simulation skills available to the agent team"),
    SlashCommand("/tools", "/tools", "show executable runtime tools, safety policy, and approval gates"),
    SlashCommand("/memory", "/memory [live]", "show GraphDB brain query plan or live read-only health"),
    SlashCommand(
        "/runtime",
        "/runtime [tools|--smoke|--tool-gateway]",
        "exercise the OpenAI Agents SDK runtime or attached-tool gateway path",
    ),
    SlashCommand(
        "/setup",
        "/setup wizard|runtime|endpoint|graphdb [...]",
        "edit saved runtime config, endpoint, GraphDB brain, and compute resources",
    ),
    SlashCommand("/status", "/status", "show session, model, ledgers, and latest agent board"),
    SlashCommand("/log", "/log [--limit N]", "show recent session events"),
    SlashCommand("/timeline", "/timeline [--limit N]", "show ordered global/session/agent/subagent event rail"),
    SlashCommand("/resume", "/resume [latest|session_id|path]", "resume a global session from inside the TUI"),
    SlashCommand("/run", "/run [--output-dir PATH] <goal>", "ask the main Orchestrator to prepare a run bundle"),
    SlashCommand("/ui", "/ui", "show the HTML controller launch command"),
    SlashCommand("/exit", "/exit", "close the interactive shell"),
)


def simulation_skill_rows() -> tuple[tuple[str, str], ...]:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_summary_rows

    return markdown_skill_summary_rows()


def markdown_skill_commands() -> tuple[SlashCommand, ...]:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_specs

    return tuple(
        SlashCommand(spec.command, f"{spec.command} <message>", f"{spec.name} skill -> {spec.agent_id}: {spec.summary}")
        for spec in markdown_skill_specs()
    )


def all_commands() -> tuple[SlashCommand, ...]:
    commands = [*COMMANDS]
    seen = {command.name for command in commands}
    for command in markdown_skill_commands():
        if command.name in seen:
            continue
        commands.append(command)
        seen.add(command.name)
    return tuple(commands)


def command_names() -> tuple[str, ...]:
    names = {command.name for command in all_commands()}
    return tuple(sorted(names))


def suggested_commands(prefix: str) -> tuple[SlashCommand, ...]:
    if prefix == "/":
        return all_commands()
    return tuple(command for command in all_commands() if command.name.startswith(prefix) or prefix in command.usage)
