from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, TextIO


@dataclass(frozen=True, slots=True)
class AnsiPalette:
    reset: str
    border: str
    title: str
    body: str
    muted: str
    accent: str
    warning: str
    success: str
    danger: str


@dataclass(frozen=True, slots=True)
class TuiTheme:
    name: str
    palette: AnsiPalette


PLAIN_PALETTE: Final = AnsiPalette("", "", "", "", "", "", "", "", "")
LAB_CONTROL_PALETTE: Final = AnsiPalette(
    reset="\033[0m",
    border="\033[38;2;51;65;59m",
    title="\033[1;38;2;99;199;178m",
    body="\033[38;2;229;235;231m",
    muted="\033[38;2;154;168;160m",
    accent="\033[38;2;99;199;178m",
    warning="\033[38;2;216;166;87m",
    success="\033[38;2;132;200;138m",
    danger="\033[38;2;224;108;117m",
)
PLAIN_THEME: Final = TuiTheme("lab-control", PLAIN_PALETTE)
COLOR_THEME: Final = TuiTheme("lab-control", LAB_CONTROL_PALETTE)


def theme_for(output_stream: TextIO) -> TuiTheme:
    if _color_enabled(output_stream):
        return COLOR_THEME
    return PLAIN_THEME


def paint(theme: TuiTheme, token: str, value: str) -> str:
    color = getattr(theme.palette, token)
    if not color:
        return value
    return f"{color}{value}{theme.palette.reset}"


def _color_enabled(output_stream: TextIO) -> bool:
    override = os.environ.get("ASA_TUI_COLOR", "").strip().lower()
    if override in {"0", "false", "no", "off"} or "NO_COLOR" in os.environ:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(output_stream, "isatty", lambda: False)())
