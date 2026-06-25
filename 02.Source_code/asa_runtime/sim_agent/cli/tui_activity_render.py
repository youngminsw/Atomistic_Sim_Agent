from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol, TextIO

from sim_agent.schemas._parse import JsonMap

from .tui_semantic import write_semantic_lines
from .tui_theme import PLAIN_THEME, TuiTheme, paint, theme_for
from .tui_width import cell_width, pad_cells, trim_cells


MIN_BOX_WIDTH: Final = 60
MAX_BOX_WIDTH: Final = 92


class ActivityResult(Protocol):
    target: str
    turn_status: str
    model_id: str
    selected_tools: tuple[str, ...]
    assistant_content: str
    runtime_events: tuple[JsonMap, ...]


@dataclass(frozen=True, slots=True)
class ActivityStyle:
    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    rule_left: str
    rule_right: str
    horizontal: str
    vertical: str
    trim_marker: str
    theme: TuiTheme
    box_width: int = MAX_BOX_WIDTH
    inner_width: int = MAX_BOX_WIDTH - 2


UNICODE_STYLE: Final = ActivityStyle("╭", "╮", "╰", "╯", "├", "┤", "─", "│", "…", PLAIN_THEME)
ASCII_STYLE: Final = ActivityStyle("+", "+", "+", "+", "|", "|", "-", "|", "...", PLAIN_THEME)


def write_activity_rail(result: ActivityResult, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    tool_names = result.selected_tools or ("none",)
    output_stream.write(_top("ASA Activity Rail", style))
    output_stream.write(_row("Agent", f"{result.target} · {result.turn_status} · {result.model_id}", style))
    output_stream.write(_row("Progress", "model stream -> tool call -> output -> transcript", style, tone="accent"))
    output_stream.write(_rule(style))
    output_stream.write(_row("Tool Call", ", ".join(tool_names), style, tone="accent"))
    output_stream.write(_row("Output", result.assistant_content, style))
    output_stream.write(_rule(style))
    for line in _visible_event_lines(result.runtime_events[-8:]):
        output_stream.write(_row("Runtime Events", line, style, tone="muted"))
    output_stream.write(_bottom(style))
    write_semantic_lines(output_stream, (*_semantic_tool_lines(tool_names), *_semantic_event_lines(result.runtime_events[-8:])))


def _visible_event_lines(events: Sequence[JsonMap]) -> tuple[str, ...]:
    if not events:
        return ("none",)
    return tuple(_event_summary(event) for event in events)


def _semantic_tool_lines(tool_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"tool_call={tool_name}" for tool_name in tool_names)


def _semantic_event_lines(events: Sequence[JsonMap]) -> tuple[str, ...]:
    if not events:
        return ("event=none",)
    return tuple(_runtime_event_line(event) for event in events)


def _runtime_event_line(event: JsonMap) -> str:
    event_type = event.get("event_type")
    details = _event_payload(event)
    status = details.get("status", "")
    detail = details.get("tool_name") or details.get("summary") or details.get("text") or details.get("role") or ""
    return f"event={event_type} status={status} detail={detail}"


def _event_summary(event: JsonMap) -> str:
    details = _event_payload(event)
    event_type = str(event.get("event_type") or "event")
    status = str(details.get("status") or "")
    detail = details.get("tool_name") or details.get("summary") or details.get("text") or details.get("role") or ""
    parts = tuple(part for part in (event_type, status, str(detail)) if part)
    return " · ".join(parts)


def _event_payload(event: JsonMap) -> JsonMap:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _top(title: str, style: ActivityStyle) -> str:
    label = f" {title} "
    right = style.horizontal * max(0, style.box_width - cell_width(label) - 2)
    return f"{_border(style, style.top_left)}{paint(style.theme, 'title', label)}{_border(style, right + style.top_right)}\n"


def _rule(style: ActivityStyle) -> str:
    return f"{_border(style, style.rule_left + (style.horizontal * style.inner_width) + style.rule_right)}\n"


def _bottom(style: ActivityStyle) -> str:
    return f"{_border(style, style.bottom_left + (style.horizontal * style.inner_width) + style.bottom_right)}\n"


def _row(left: str, right: str, style: ActivityStyle, *, tone: str = "body") -> str:
    body_width = max(10, style.inner_width - 2)
    left_width = max(16, min(24, body_width // 3))
    right_width = max(8, body_width - left_width - 1)
    left_text = pad_cells(trim_cells(left, left_width, style.trim_marker), left_width)
    right_text = pad_cells(trim_cells(right, right_width, style.trim_marker), right_width)
    return f"{_border(style, style.vertical)} {paint(style.theme, tone, left_text + ' ' + right_text)} {_border(style, style.vertical)}\n"


def _border(style: ActivityStyle, value: str) -> str:
    return paint(style.theme, "border", value)


def _style_for(output_stream: TextIO) -> ActivityStyle:
    encoding = output_stream.encoding or "utf-8"
    box_width = _box_width_for(output_stream)
    try:
        "╭─╮│╰╯├┤…".encode(encoding)
    except UnicodeEncodeError:
        return replace(ASCII_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    except LookupError:
        return replace(ASCII_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    return replace(UNICODE_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)


def _box_width_for(output_stream: TextIO) -> int:
    fallback_width = MAX_BOX_WIDTH if not output_stream.isatty() else 100
    terminal_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return min(MAX_BOX_WIDTH, max(MIN_BOX_WIDTH, terminal_width))
