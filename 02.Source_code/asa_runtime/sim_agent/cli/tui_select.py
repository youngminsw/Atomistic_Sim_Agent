from __future__ import annotations

import os
import select
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TextIO


UP_KEY: Final = "\x1b[A"
DOWN_KEY: Final = "\x1b[B"
ESC_KEY: Final = "\x1b"
CTRL_C_KEY: Final = "\x03"
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
    selected = 0
    rendered_lines = len(options) + 2
    fd = input_stream.fileno()
    import termios
    import tty

    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        output_stream.write(f"{title}\n")
        output_stream.write("Use ↑/↓ then Enter. Esc cancels.\n")
        output_stream.flush()
        output_stream.write("\x1b[?25l")
        _render_options(options, selected, output_stream)
        while True:
            key = _read_key_from_fd(fd)
            match key:  # noqa: MATCH_OK - terminal input is an open string stream.
                case "\x1b[A":
                    selected = (selected - 1) % len(options)
                case "\x1b[B":
                    selected = (selected + 1) % len(options)
                case "\r" | "\n":
                    _clear_menu(rendered_lines, output_stream)
                    return options[selected].value
                case "\x1b":
                    _clear_menu(rendered_lines, output_stream)
                    return None
                case "":
                    _clear_menu(rendered_lines, output_stream)
                    return None
                case "\x03":
                    _clear_menu(rendered_lines, output_stream)
                    raise KeyboardInterrupt
                case _:
                    pass
            _move_up(len(options), output_stream)
            _render_options(options, selected, output_stream)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        output_stream.write("\x1b[?25h")
        output_stream.flush()


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
    width = _terminal_width()
    for index, option in enumerate(options):
        marker = "❯" if index == selected else " "
        line = f"{marker} {option.label:<18} {option.summary}"
        output_stream.write(f"\r\x1b[2K{line[:width]}\n")
    output_stream.flush()


def _move_up(lines: int, output_stream: TextIO) -> None:
    output_stream.write(f"\x1b[{lines}A\r")


def _clear_menu(lines: int, output_stream: TextIO) -> None:
    _move_up(lines, output_stream)
    for index in range(lines):
        output_stream.write("\r\x1b[2K")
        if index < lines - 1:
            output_stream.write("\n")
    if lines > 1:
        _move_up(lines - 1, output_stream)
    output_stream.flush()


def _read_key_from_fd(fd: int) -> str:
    try:
        first = os.read(fd, 1)
    except OSError:
        return ""
    if not first:
        return ""
    if first != ESC_KEY.encode():
        return first.decode("utf-8", errors="ignore")
    if not _fd_ready(fd, timeout_s=0.15):
        return ESC_KEY
    buffer = bytearray(first)
    while _fd_ready(fd, timeout_s=0.03):
        try:
            buffer.extend(os.read(fd, 1))
        except OSError:
            return ESC_KEY
        if len(buffer) >= 3 and buffer[1:2] == b"[" and buffer[-1] in b"ABCD":
            break
        if len(buffer) >= 8:
            break
    return _decode_key_bytes(bytes(buffer))


def _decode_key_bytes(value: bytes) -> str:
    if value == b"\x1b[A":
        return UP_KEY
    if value == b"\x1b[B":
        return DOWN_KEY
    if value == b"\x03":
        return CTRL_C_KEY
    if value in {b"\r", b"\n"}:
        return value.decode("ascii")
    if value.startswith(b"\x1b"):
        return ESC_KEY
    return value.decode("utf-8", errors="ignore")


def _terminal_width() -> int:
    return max(20, shutil.get_terminal_size(fallback=(100, 24)).columns)


def _fd_ready(fd: int, timeout_s: float = 0.05) -> bool:
    try:
        readable, _writable, _error = select.select((fd,), (), (), timeout_s)
    except (OSError, ValueError):
        return False
    return bool(readable)


def _supports_posix_raw(input_stream: TextIO) -> bool:
    return os.name == "posix" and input_stream.isatty()
