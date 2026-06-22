from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TextIO

from .tui_select import MenuOption, choose_option


THINKING_LEVELS: Final[tuple[str, ...]] = ("high", "xhigh", "medium", "low", "minimal", "off", "inherit", "max")
THINKING_SUMMARY_BY_LEVEL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "high": "deep planning and high-stakes control",
        "xhigh": "maximum reasoning for hardest planning",
        "medium": "balanced subagent work",
        "low": "fast lightweight subtasks",
        "minimal": "short execution lanes and mechanical edits",
        "off": "disable explicit thinking budget when the provider allows it",
        "inherit": "use the parent or profile default",
        "max": "highest available reasoning tier",
    }
)


def choose_thinking_level(
    title: str,
    default_level: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str | None:
    return choose_option(
        title,
        tuple(
            MenuOption(level, _thinking_label(level, default_level), _thinking_summary(level))
            for level in _ordered_levels(default_level)
        ),
        input_stream,
        output_stream,
    )


def is_thinking_level(value: str) -> bool:
    return value in THINKING_LEVELS


def thinking_levels_text() -> str:
    return ",".join(THINKING_LEVELS)


def _ordered_levels(default_level: str) -> Sequence[str]:
    if default_level not in THINKING_LEVELS:
        return THINKING_LEVELS
    return (default_level, *(level for level in THINKING_LEVELS if level != default_level))


def _thinking_label(level: str, default_level: str) -> str:
    if level == default_level:
        return f"{level} (default)"
    return level


def _thinking_summary(level: str) -> str:
    return THINKING_SUMMARY_BY_LEVEL.get(level, "custom reasoning effort")
