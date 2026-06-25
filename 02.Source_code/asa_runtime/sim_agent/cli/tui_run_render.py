from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Protocol, TextIO

from .tui_paths import display_path
from .tui_semantic import write_semantic_lines
from .tui_theme import PLAIN_THEME, TuiTheme, paint, theme_for
from .tui_width import cell_width, pad_cells, trim_cells


MIN_BOX_WIDTH: Final = 60
MAX_BOX_WIDTH: Final = 92


class RunRailReport(Protocol):
    run_id: str
    artifact_dir: str
    ledger_path: object


@dataclass(frozen=True, slots=True)
class RunRailStyle:
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


UNICODE_STYLE: Final = RunRailStyle("╭", "╮", "╰", "╯", "├", "┤", "─", "│", "…", PLAIN_THEME)
ASCII_STYLE: Final = RunRailStyle("+", "+", "+", "+", "|", "|", "-", "|", "...", PLAIN_THEME)


def write_run_rail(
    *,
    goal: str,
    provider: str,
    model: str,
    reasoning_effort: str,
    report: RunRailReport,
    output_stream: TextIO,
) -> None:
    style = _style_for(output_stream)
    output_stream.write(_top("ASA Run Rail", style))
    output_stream.write(_row("Orchestrator", f"prepared · {provider}/{model} · {reasoning_effort}", style))
    output_stream.write(_row("Progress", "request parsed -> gates staged -> ledger written", style, tone="accent"))
    output_stream.write(_rule(style))
    output_stream.write(_row("Goal", goal, style))
    output_stream.write(_row("Output", f"run bundle prepared · {report.run_id}", style))
    output_stream.write(_row("Artifacts", display_path(Path(str(report.ledger_path))), style, tone="muted"))
    output_stream.write(_row("Next", "/hud · /timeline · /ui controller", style, tone="accent"))
    output_stream.write(_bottom(style))
    write_semantic_lines(
        output_stream,
        (
            "run_progress=prepared",
            f"run_card=true run_id={report.run_id}",
            f"run_artifact_ledger={report.ledger_path}",
        ),
    )


def _top(title: str, style: RunRailStyle) -> str:
    label = f" {title} "
    right = style.horizontal * max(0, style.box_width - cell_width(label) - 2)
    return f"{_border(style, style.top_left)}{paint(style.theme, 'title', label)}{_border(style, right + style.top_right)}\n"


def _rule(style: RunRailStyle) -> str:
    return f"{_border(style, style.rule_left + (style.horizontal * style.inner_width) + style.rule_right)}\n"


def _bottom(style: RunRailStyle) -> str:
    return f"{_border(style, style.bottom_left + (style.horizontal * style.inner_width) + style.bottom_right)}\n"


def _row(left: str, right: str, style: RunRailStyle, *, tone: str = "body") -> str:
    body_width = max(10, style.inner_width - 2)
    left_width = max(16, min(24, body_width // 3))
    right_width = max(8, body_width - left_width - 1)
    left_text = pad_cells(trim_cells(left, left_width, style.trim_marker), left_width)
    right_text = pad_cells(trim_cells(right, right_width, style.trim_marker), right_width)
    return f"{_border(style, style.vertical)} {paint(style.theme, tone, left_text + ' ' + right_text)} {_border(style, style.vertical)}\n"


def _border(style: RunRailStyle, value: str) -> str:
    return paint(style.theme, "border", value)


def _style_for(output_stream: TextIO) -> RunRailStyle:
    encoding = output_stream.encoding or "utf-8"
    box_width = _box_width_for(output_stream)
    try:
        "╭─╮│╰╯├┤…".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return replace(ASCII_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)
    return replace(UNICODE_STYLE, theme=theme_for(output_stream), box_width=box_width, inner_width=box_width - 2)


def _box_width_for(output_stream: TextIO) -> int:
    fallback_width = MAX_BOX_WIDTH if not output_stream.isatty() else 100
    terminal_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return min(MAX_BOX_WIDTH, max(MIN_BOX_WIDTH, terminal_width))
