from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TextIO


UP_KEY: Final = "\x1b[A"
DOWN_KEY: Final = "\x1b[B"
ESC_KEY: Final = "\x1b"
ENTER_KEYS: Final = ("\r", "\n")


@dataclass(frozen=True, slots=True)
class MenuOption:
    value: str
    label: str
    summary: str


def choose_option(title: str, options: Sequence[MenuOption], input_stream: TextIO, output_stream: TextIO) -> str | None:
    if not options:
        return None
    if not _supports_posix_raw(input_stream):
        return _choose_option_by_line(title, options, input_stream, output_stream)
    import termios
    import tty

    selected = 0
    output_stream.write(f"\n{title}\n")
    output_stream.write("Use ↑/↓ then Enter. Esc cancels.\n")
    output_stream.flush()
    _render_options(options, selected, output_stream)
    fd = input_stream.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = _read_key(input_stream)
            match key:
                case "\x1b[A":
                    selected = (selected - 1) % len(options)
                case "\x1b[B":
                    selected = (selected + 1) % len(options)
                case "\r" | "\n":
                    output_stream.write("\n")
                    output_stream.flush()
                    return options[selected].value
                case "\x1b":
                    output_stream.write("\n")
                    output_stream.flush()
                    return None
                case _:
                    pass
            _move_up(len(options), output_stream)
            _render_options(options, selected, output_stream)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def prompt_visible(label: str, default: str, input_stream: TextIO, output_stream: TextIO) -> str:
    output_stream.write(f"{label} [{default}]: ")
    output_stream.flush()
    line = input_stream.readline().strip()
    return line or default


def prompt_secret(label: str, input_stream: TextIO, output_stream: TextIO) -> str:
    output_stream.write(f"{label}: ")
    output_stream.flush()
    if not _supports_posix_raw(input_stream):
        return input_stream.readline().strip()
    import termios

    fd = input_stream.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        attrs = termios.tcgetattr(fd)
        attrs[3] = attrs[3] & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        return input_stream.readline().strip()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        output_stream.write("\n")
        output_stream.flush()


def _choose_option_by_line(
    title: str,
    options: Sequence[MenuOption],
    input_stream: TextIO,
    output_stream: TextIO,
) -> str | None:
    output_stream.write(f"\n{title}\n")
    for index, option in enumerate(options, start=1):
        output_stream.write(f"{index}. {option.label} - {option.summary}\n")
    output_stream.write("Select number, label, or value. Empty selects the first option: ")
    output_stream.flush()
    raw = input_stream.readline().strip()
    if not raw:
        return options[0].value
    if raw.isdecimal():
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index].value
    normalized = raw.casefold()
    for option in options:
        if normalized in {option.value.casefold(), option.label.casefold()}:
            return option.value
    return None


def _render_options(options: Sequence[MenuOption], selected: int, output_stream: TextIO) -> None:
    for index, option in enumerate(options):
        marker = "❯" if index == selected else " "
        output_stream.write(f"\x1b[2K{marker} {option.label:<18} {option.summary}\n")
    output_stream.flush()


def _move_up(lines: int, output_stream: TextIO) -> None:
    output_stream.write(f"\x1b[{lines}A")


def _read_key(input_stream: TextIO) -> str:
    first = input_stream.read(1)
    if first != ESC_KEY:
        return first
    second = input_stream.read(1)
    if second != "[":
        return ESC_KEY
    third = input_stream.read(1)
    return f"{ESC_KEY}[{third}"


def _supports_posix_raw(input_stream: TextIO) -> bool:
    return os.name == "posix" and input_stream.isatty()
