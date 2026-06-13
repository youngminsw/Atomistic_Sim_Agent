from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES

from .tui_catalog import COMMANDS, SIMULATION_SKILLS, SlashCommand
from .tui_state import TuiState


BOX_WIDTH: Final = 92
INNER_WIDTH: Final = BOX_WIDTH - 2


@dataclass(frozen=True, slots=True)
class AgentStatusRow:
    agent_id: str
    status: str
    activity: str
    peer: str = ""
    heartbeat_s: int | None = None


def write_welcome(state: TuiState, output_stream: TextIO) -> None:
    output_stream.write(_top("Atomistic Simulation Agent"))
    output_stream.write(_row("Welcome back to the simulation agent shell.", "Tips"))
    output_stream.write(_row("▗▄▖  ASA controls MD, MDN, Level-Set, GraphDB, and QA.", "Type / for palette"))
    output_stream.write(_row(f"Model {state.model.provider}/{state.model.name} · {state.model.auth_mode}", "/model status"))
    output_stream.write(_row(f"Session {state.session_id}", "/agents /team /runtime"))
    output_stream.write(_row(_trim_path(state.session_dir), "/ui for controller"))
    output_stream.write(_rule())
    output_stream.write(_row("Plain text is routed to the main Orchestrator as a run goal.", "/help for commands"))
    output_stream.write(_bottom())
    write_agent_workboard("Agent Workboard", initial_agent_rows(), output_stream)


def write_help_panel(output_stream: TextIO) -> None:
    output_stream.write(_top("Slash Command Palette"))
    for command in COMMANDS:
        output_stream.write(_row(command.usage, command.summary))
    output_stream.write(_rule())
    output_stream.write(_row("simulation skills", ", ".join(name for name, _summary in SIMULATION_SKILLS)))
    output_stream.write(_bottom())


def write_command_palette(prefix: str, output_stream: TextIO) -> None:
    commands = _palette_commands(prefix)
    output_stream.write(_top("Slash Command Palette"))
    for command in commands:
        output_stream.write(_row(command.usage, command.summary))
    output_stream.write(_rule())
    output_stream.write(_row("simulation skills", ", ".join(name for name, _summary in SIMULATION_SKILLS)))
    output_stream.write(_bottom())


def write_agent_workboard(title: str, rows: Sequence[AgentStatusRow], output_stream: TextIO) -> None:
    output_stream.write(_top(title))
    output_stream.write(_row("agent", "status · current work"))
    output_stream.write(_rule())
    for row in rows:
        detail = _agent_detail(row)
        output_stream.write(_row(row.agent_id, detail))
    output_stream.write(_bottom())


def initial_agent_rows() -> tuple[AgentStatusRow, ...]:
    rows = [AgentStatusRow("orchestrator", "idle", "ready to route user goals")]
    rows.extend(AgentStatusRow(role.role_id, "idle", role.boundary) for role in AGENT_ROLES)
    return tuple(rows)


def _palette_commands(prefix: str) -> tuple[SlashCommand, ...]:
    matches = tuple(command for command in COMMANDS if command.name.startswith(prefix) or prefix in command.usage)
    if matches:
        return matches
    return COMMANDS


def _agent_detail(row: AgentStatusRow) -> str:
    peer = f" -> {row.peer}" if row.peer else ""
    heartbeat = f" · heartbeat {row.heartbeat_s}s" if row.heartbeat_s is not None else ""
    return f"{row.status}{peer} · {row.activity}{heartbeat}"


def _top(title: str) -> str:
    label = f" {title} "
    right = "─" * max(0, BOX_WIDTH - len(label) - 2)
    return f"╭{label}{right}╮\n"


def _rule() -> str:
    return f"├{'─' * INNER_WIDTH}┤\n"


def _bottom() -> str:
    return f"╰{'─' * INNER_WIDTH}╯\n"


def _row(left: str, right: str = "") -> str:
    left_text = _trim(left, 48)
    right_text = _trim(right, 37)
    body = f"{left_text:<50}{right_text:<38}"
    return f"│ {body[:INNER_WIDTH - 2]} │\n"


def _trim(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return f"{value[: max(0, width - 1)]}…"


def _trim_path(path: Path) -> str:
    parts = path.parts
    if len(parts) <= 4:
        return str(path)
    return str(Path("…", *parts[-3:]))
