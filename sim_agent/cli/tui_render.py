from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES

from .tui_catalog import COMMANDS, SIMULATION_SKILLS, SlashCommand
from .tui_hud import build_hud_status, ledger_label
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


@dataclass(frozen=True, slots=True)
class BoxStyle:
    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    rule_left: str
    rule_right: str
    horizontal: str
    vertical: str
    trim_marker: str
    logo: str
    separator: str


UNICODE_BOX: Final = BoxStyle("╭", "╮", "╰", "╯", "├", "┤", "─", "│", "…", "▗▄▖", " · ")
ASCII_BOX: Final = BoxStyle("+", "+", "+", "+", "|", "|", "-", "|", "...", "ASA", " - ")


def write_welcome(state: TuiState, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    hud = build_hud_status(state)
    connection = "connected" if hud.connected else "not connected"
    output_stream.write(_top("Atomistic Simulation Agent", style))
    output_stream.write(_row("Welcome back to the simulation agent shell.", "Tips", style))
    output_stream.write(_row(f"{style.logo}  ASA controls MD, MDN, Level-Set, GraphDB, and QA.", "Type / for palette", style))
    output_stream.write(_row(f"HUD {hud.provider}/{hud.model}{style.separator}{hud.auth_mode}", connection, style))
    if not hud.connected:
        output_stream.write(_row("Model is not connected. Run /login.", "/model set", style))
    output_stream.write(_row(f"Session {state.session_id}", "/agents /team /runtime", style))
    output_stream.write(_row(_trim_path(state.session_dir, style), "/ui for controller", style))
    output_stream.write(_rule(style))
    output_stream.write(_row("Plain text is routed to the main Orchestrator as a run goal.", "/help for commands", style))
    output_stream.write(_bottom(style))
    write_agent_workboard("Agent Workboard", initial_agent_rows(), output_stream)


def write_hud_panel(state: TuiState, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    hud = build_hud_status(state)
    connected = "connected" if hud.connected else "not connected"
    output_stream.write(_top("ASA HUD", style))
    output_stream.write(_row("Model", f"{hud.provider}/{hud.model}", style))
    output_stream.write(_row("Auth", f"{hud.auth_mode}{style.separator}{connected}", style))
    output_stream.write(_row("Session", hud.session_id, style))
    output_stream.write(_row("Run ledger", ledger_label(hud.last_run_ledger), style))
    output_stream.write(_row("Team ledger", ledger_label(hud.team_ledger), style))
    output_stream.write(_row("Runtime ledger", ledger_label(hud.runtime_ledger), style))
    output_stream.write(_rule(style))
    output_stream.write(_row(hud.friendly_message, hud.action_hint, style))
    output_stream.write(_bottom(style))
    output_stream.write("hud=true\n")
    output_stream.write(f"model_connected={hud.connected}\n")
    output_stream.write(f"provider={hud.provider} model={hud.model} auth_mode={hud.auth_mode}\n")
    output_stream.write(f"connection_label={hud.connection_label}\n")
    output_stream.write(f"credential_store={hud.credential_store}\n")
    output_stream.write(f"model_notice={hud.friendly_message}\n")
    output_stream.write(f"model_action={hud.action_hint}\n")


def write_help_panel(output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    output_stream.write(_top("Slash Command Palette", style))
    for command in COMMANDS:
        output_stream.write(_row(command.usage, command.summary, style))
    output_stream.write(_rule(style))
    output_stream.write(_row("simulation skills", ", ".join(name for name, _summary in SIMULATION_SKILLS), style))
    output_stream.write(_bottom(style))


def write_command_palette(prefix: str, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    commands = _palette_commands(prefix)
    output_stream.write(_top("Slash Command Palette", style))
    for command in commands:
        output_stream.write(_row(command.usage, command.summary, style))
    output_stream.write(_rule(style))
    output_stream.write(_row("simulation skills", ", ".join(name for name, _summary in SIMULATION_SKILLS), style))
    output_stream.write(_bottom(style))


def write_agent_workboard(title: str, rows: Sequence[AgentStatusRow], output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    output_stream.write(_top(title, style))
    output_stream.write(_row("agent", f"status{style.separator}current work", style))
    output_stream.write(_rule(style))
    for row in rows:
        detail = _agent_detail(row, style)
        output_stream.write(_row(row.agent_id, detail, style))
    output_stream.write(_bottom(style))


def initial_agent_rows() -> tuple[AgentStatusRow, ...]:
    rows = [AgentStatusRow("orchestrator", "idle", "ready to route user goals")]
    rows.extend(AgentStatusRow(role.role_id, "idle", role.boundary) for role in AGENT_ROLES)
    return tuple(rows)


def _palette_commands(prefix: str) -> tuple[SlashCommand, ...]:
    matches = tuple(command for command in COMMANDS if command.name.startswith(prefix) or prefix in command.usage)
    if matches:
        return matches
    return COMMANDS


def _agent_detail(row: AgentStatusRow, style: BoxStyle) -> str:
    peer = f" -> {row.peer}" if row.peer else ""
    heartbeat = f"{style.separator}heartbeat {row.heartbeat_s}s" if row.heartbeat_s is not None else ""
    return f"{row.status}{peer}{style.separator}{row.activity}{heartbeat}"


def _top(title: str, style: BoxStyle) -> str:
    label = f" {title} "
    right = style.horizontal * max(0, BOX_WIDTH - len(label) - 2)
    return f"{style.top_left}{label}{right}{style.top_right}\n"


def _rule(style: BoxStyle) -> str:
    return f"{style.rule_left}{style.horizontal * INNER_WIDTH}{style.rule_right}\n"


def _bottom(style: BoxStyle) -> str:
    return f"{style.bottom_left}{style.horizontal * INNER_WIDTH}{style.bottom_right}\n"


def _row(left: str, right: str, style: BoxStyle) -> str:
    left_text = _trim(left, 48, style)
    right_text = _trim(right, 37, style)
    body = f"{left_text:<50}{right_text:<38}"
    return f"{style.vertical} {body[:INNER_WIDTH - 2]} {style.vertical}\n"


def _trim(value: str, width: int, style: BoxStyle) -> str:
    if len(value) <= width:
        return value
    marker_width = len(style.trim_marker)
    return f"{value[: max(0, width - marker_width)]}{style.trim_marker}"


def _trim_path(path: Path, style: BoxStyle) -> str:
    parts = path.parts
    if len(parts) <= 4:
        return str(path)
    return str(Path(style.trim_marker, *parts[-3:]))


def _style_for(output_stream: TextIO) -> BoxStyle:
    encoding = output_stream.encoding or "utf-8"
    try:
        "╭─╮│╰╯├┤▗▄▖…".encode(encoding)
    except UnicodeEncodeError:
        return ASCII_BOX
    except LookupError:
        return ASCII_BOX
    return UNICODE_BOX
