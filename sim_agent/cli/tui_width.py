from __future__ import annotations

import unicodedata
from typing import Final


WIDE_EAST_ASIAN_WIDTHS: Final[frozenset[str]] = frozenset({"F", "W"})


def cell_width(value: str) -> int:
    width = 0
    for char in value:
        width += 2 if unicodedata.east_asian_width(char) in WIDE_EAST_ASIAN_WIDTHS else 1
    return width


def trim_cells(value: str, width: int, trim_marker: str) -> str:
    if cell_width(value) <= width:
        return value
    marker_width = cell_width(trim_marker)
    budget = max(0, width - marker_width)
    used = 0
    chars: list[str] = []
    for char in value:
        next_width = cell_width(char)
        if used + next_width > budget:
            break
        chars.append(char)
        used += next_width
    return f"{''.join(chars)}{trim_marker}"


def pad_cells(value: str, width: int) -> str:
    return f"{value}{' ' * max(0, width - cell_width(value))}"
