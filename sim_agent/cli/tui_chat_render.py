from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Protocol, TextIO

from .tui_paths import display_path
from .tui_semantic import write_semantic_line, write_semantic_lines
from .tui_theme import PLAIN_THEME, TuiTheme, paint, theme_for
from .tui_width import cell_width, pad_cells, trim_cells


MIN_BOX_WIDTH: Final = 60
MAX_BOX_WIDTH: Final = 92


class ChatRenderMessage(Protocol):
    role: str
    target: str | None
    content: str


class ChatRenderView(Protocol):
    messages: Sequence[ChatRenderMessage]
    message_count: int
    corrupt_lines: int
    path: Path


@dataclass(frozen=True, slots=True)
class ChatBoxStyle:
    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    horizontal: str
    vertical: str
    trim_marker: str
    theme: TuiTheme
    box_width: int = MAX_BOX_WIDTH
    inner_width: int = MAX_BOX_WIDTH - 2


UNICODE_STYLE: Final = ChatBoxStyle("╭", "╮", "╰", "╯", "─", "│", "…", PLAIN_THEME)
ASCII_STYLE: Final = ChatBoxStyle("+", "+", "+", "+", "-", "|", "...", PLAIN_THEME)


def write_chat_deck(view: ChatRenderView, output_stream: TextIO) -> None:
    style = _style_for(output_stream)
    output_stream.write(_top("ASA Chat Deck", style))
    output_stream.write(_row("target", "message", style, tone="accent"))
    output_stream.write(_rule(style))
    if not view.messages:
        output_stream.write(_row("orchestrator", "No chat yet. Type a goal or @md_agent <task>.", style, tone="muted"))
    for message in view.messages:
        output_stream.write(_row(f"{message.role}@{message.target or 'orchestrator'}", message.content, style))
    output_stream.write(_rule(style))
    output_stream.write(_row("palette", "/ commands · @ agents · /hud status", style, tone="accent"))
    output_stream.write(_bottom(style))
    write_semantic_lines(
        output_stream,
        (
            "chat_window=true",
            f"chat_message_count={view.message_count}",
            f"chat_transcript_path={display_path(view.path)}",
            f"chat_transcript_corrupt_lines={view.corrupt_lines}",
        ),
    )
    if view.messages:
        active_target = next(
            (message.target for message in reversed(view.messages) if message.target and message.target != "orchestrator"),
            "orchestrator",
        )
        write_semantic_line(output_stream, f"chat_target={active_target}")


def _top(title: str, style: ChatBoxStyle) -> str:
    label = f" {title} "
    right = style.horizontal * max(0, style.box_width - cell_width(label) - 2)
    return f"{_border(style, style.top_left)}{paint(style.theme, 'title', label)}{_border(style, right + style.top_right)}\n"


def _rule(style: ChatBoxStyle) -> str:
    return f"{_border(style, style.vertical + (style.horizontal * style.inner_width) + style.vertical)}\n"


def _bottom(style: ChatBoxStyle) -> str:
    return f"{_border(style, style.bottom_left + (style.horizontal * style.inner_width) + style.bottom_right)}\n"


def _row(left: str, right: str, style: ChatBoxStyle, *, tone: str = "body") -> str:
    body_width = max(10, style.inner_width - 2)
    left_width = max(18, min(28, body_width // 3))
    right_width = max(8, body_width - left_width - 1)
    left_text = pad_cells(trim_cells(left, left_width, style.trim_marker), left_width)
    right_text = pad_cells(trim_cells(right, right_width, style.trim_marker), right_width)
    body = f"{left_text} {right_text}"
    return f"{_border(style, style.vertical)} {paint(style.theme, tone, body)} {_border(style, style.vertical)}\n"


def _border(style: ChatBoxStyle, value: str) -> str:
    return paint(style.theme, "border", value)


def _style_for(output_stream: TextIO) -> ChatBoxStyle:
    encoding = output_stream.encoding or "utf-8"
    box_width = _box_width_for(output_stream)
    try:
        "╭─╮│…".encode(encoding)
    except UnicodeEncodeError:
        return replace(ASCII_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    except LookupError:
        return replace(ASCII_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    return replace(UNICODE_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)


def _box_width_for(output_stream: TextIO) -> int:
    fallback_width = MAX_BOX_WIDTH if not output_stream.isatty() else 100
    terminal_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return min(MAX_BOX_WIDTH, max(MIN_BOX_WIDTH, terminal_width))
