from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES

from .tui_catalog import COMMANDS, SIMULATION_SKILLS, SlashCommand
from .tui_chat import chat_hud_summary
from .tui_hud import build_hud_status, ledger_label
from .tui_state import TuiState
from .tui_theme import PLAIN_THEME, TuiTheme, paint, theme_for
from .tui_width import cell_width, pad_cells, trim_cells


MIN_BOX_WIDTH: Final = 60
MAX_BOX_WIDTH: Final = 92


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
    theme: TuiTheme
    box_width: int = MAX_BOX_WIDTH
    inner_width: int = MAX_BOX_WIDTH - 2


UNICODE_BOX: Final = BoxStyle("╭", "╮", "╰", "╯", "├", "┤", "─", "│", "…", "▗▄▖", " · ", PLAIN_THEME)
ASCII_BOX: Final = BoxStyle("+", "+", "+", "+", "|", "|", "-", "|", "...", "ASA", " - ", PLAIN_THEME)


def write_welcome(state: TuiState, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    hud = build_hud_status(state)
    connection = "connected" if hud.connected else "login required"
    connection_tone = "success" if hud.connected else "warning"
    output_stream.write(_top("ASA Control Room", style))
    output_stream.write(_row("Atomistic Simulation Agent", "lab-control theme", style, tone="accent"))
    output_stream.write(_row("runtime console", "MD · MDN · Level-Set · GraphDB · QA", style))
    output_stream.write(_row(f"{style.logo} model rail", f"{hud.provider}/{hud.model}{style.separator}{hud.auth_mode}{style.separator}{connection}", style, tone=connection_tone))
    output_stream.write(_row("profile rail", _profile_label(hud.active_profile, hud.profile_customized), style))
    if not hud.connected:
        output_stream.write(_row("next action", "Model is not connected. Run /login; then /model set.", style, tone="warning"))
    output_stream.write(_row("evidence rail", "login -> model -> guard evidence -> run ledger", style))
    output_stream.write(_row("agent rail", "orchestrator chat + direct @agent summons", style))
    output_stream.write(_row(f"session {state.session_id}", "/agents /team /runtime", style, tone="muted"))
    output_stream.write(_row(_trim_path(state.session_dir, style), "/ui for controller", style, tone="muted"))
    output_stream.write(_rule(style))
    output_stream.write(_row("plain goal opens Orchestrator chat", "@agent routes a bounded specialist", style, tone="accent"))
    output_stream.write(_bottom(style))
    write_agent_workboard("Agent Workboard", initial_agent_rows(), output_stream)
    output_stream.write("초보자 안내: 처음이면 /guide 또는 /start, 아니면 목표를 문장으로 입력하세요.\n")


def write_hud_panel(state: TuiState, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    hud = build_hud_status(state)
    chat = chat_hud_summary(state)
    connected = "connected" if hud.connected else "not connected"
    output_stream.write(_top("ASA HUD", style))
    output_stream.write(_row("model rail", f"{hud.provider}/{hud.model}", style, tone="accent"))
    output_stream.write(_row("profile rail", _profile_label(hud.active_profile, hud.profile_customized), style))
    output_stream.write(_row("auth rail", f"{hud.auth_mode}{style.separator}{connected}", style, tone="success" if hud.connected else "warning"))
    output_stream.write(_row("session rail", hud.session_id, style, tone="muted"))
    output_stream.write(_row("chat rail", f"{chat.message_count} messages{style.separator}last {chat.last_role}@{chat.last_target}", style))
    output_stream.write(_row("activity rail", _agent_activity_label(state), style))
    output_stream.write(_row("palette rail", "/ commands, direct @agent summon", style))
    output_stream.write(_row("agent rail", "orchestrator + MD/MDN/feature/GraphDB/QA", style))
    output_stream.write(_row("run ledger", ledger_label(hud.last_run_ledger), style))
    output_stream.write(_row("team ledger", ledger_label(hud.team_ledger), style))
    output_stream.write(_row("runtime ledger", ledger_label(hud.runtime_ledger), style))
    output_stream.write(_rule(style))
    output_stream.write(_row(hud.friendly_message, hud.action_hint, style, tone="warning" if not hud.connected else "success"))
    output_stream.write(_bottom(style))
    output_stream.write("hud=true\n")
    output_stream.write("control_room_hud=true\n")
    output_stream.write(f"model_connected={hud.connected}\n")
    output_stream.write(f"active_profile={hud.active_profile}\n")
    output_stream.write(f"profile_customized={str(hud.profile_customized).lower()}\n")
    output_stream.write(f"model_profile={hud.active_profile} customized={str(hud.profile_customized).lower()}\n")
    output_stream.write(f"provider={hud.provider} model={hud.model} auth_mode={hud.auth_mode}\n")
    output_stream.write(f"connection_label={hud.connection_label}\n")
    output_stream.write(f"chat_message_count={chat.message_count}\n")
    output_stream.write(f"chat_transcript_path={chat.path}\n")
    output_stream.write(f"chat_transcript_corrupt_lines={chat.corrupt_lines}\n")
    output_stream.write(f"agent_activity_label={_agent_activity_label(state)}\n")
    output_stream.write("agent_mention_surface=true\n")
    output_stream.write("direct_agent_mention=true\n")
    output_stream.write("palette_hints=slash,at-agent,chat,hud\n")
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
    output_stream.write(_row("native agent", f"status{style.separator}current work", style, tone="accent"))
    output_stream.write(_rule(style))
    for row in rows:
        detail = _agent_detail(row, style)
        output_stream.write(_row(row.agent_id, detail, style, tone="muted" if row.status == "idle" else "accent"))
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


def _profile_label(active_profile: str, customized: bool) -> str:
    suffix = "customized" if customized else "preset"
    return f"{active_profile} · {suffix}"


def _top(title: str, style: BoxStyle) -> str:
    label = f" {title} "
    right = style.horizontal * max(0, style.box_width - cell_width(label) - 2)
    return f"{_border(style, style.top_left)}{paint(style.theme, 'title', label)}{_border(style, right + style.top_right)}\n"


def _rule(style: BoxStyle) -> str:
    return f"{_border(style, style.rule_left + (style.horizontal * style.inner_width) + style.rule_right)}\n"


def _bottom(style: BoxStyle) -> str:
    return f"{_border(style, style.bottom_left + (style.horizontal * style.inner_width) + style.bottom_right)}\n"


def _row(left: str, right: str, style: BoxStyle, *, tone: str = "body") -> str:
    body_width = max(10, style.inner_width - 2)
    left_width = max(18, min(50, body_width // 2 + 6))
    right_width = max(8, body_width - left_width)
    left_text = pad_cells(trim_cells(left, max(1, left_width - 2), style.trim_marker), left_width)
    right_text = pad_cells(trim_cells(right, max(1, right_width), style.trim_marker), right_width)
    body = trim_cells(f"{left_text}{right_text}", body_width, style.trim_marker)
    return f"{_border(style, style.vertical)} {paint(style.theme, tone, pad_cells(body, body_width))} {_border(style, style.vertical)}\n"


def _border(style: BoxStyle, value: str) -> str:
    return paint(style.theme, "border", value)


def _trim(value: str, width: int, style: BoxStyle) -> str:
    return trim_cells(value, width, style.trim_marker)


def _trim_path(path: Path, style: BoxStyle) -> str:
    parts = path.parts
    if len(parts) <= 4:
        return str(path)
    return str(Path(style.trim_marker, *parts[-3:]))


def _agent_activity_label(state: TuiState) -> str:
    from .tui_agent_activity import agent_activity_hud_label

    return agent_activity_hud_label(state)


def _style_for(output_stream: TextIO) -> BoxStyle:
    encoding = output_stream.encoding or "utf-8"
    box_width = _box_width_for(output_stream)
    try:
        "╭─╮│╰╯├┤▗▄▖…".encode(encoding)
    except UnicodeEncodeError:
        return replace(ASCII_BOX, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    except LookupError:
        return replace(ASCII_BOX, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    return replace(UNICODE_BOX, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)


def _box_width_for(output_stream: TextIO) -> int:
    fallback_width = MAX_BOX_WIDTH if not output_stream.isatty() else 100
    terminal_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return min(MAX_BOX_WIDTH, max(MIN_BOX_WIDTH, terminal_width))
