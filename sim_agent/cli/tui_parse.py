from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedLine:
    command: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedSlash:
    options: dict[str, str]
    flags: tuple[str, ...]
    remainder: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParseError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def parse_line(line: str) -> ParsedLine | None:
    if not line:
        return None
    try:
        parts = tuple(shlex.split(line))
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    return ParsedLine(command=parts[0], args=parts[1:])


def parse_options(args: Sequence[str]) -> ParsedSlash:
    options: dict[str, str] = {}
    flags: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(args):
        part = args[index]
        if part.startswith("--") and index + 1 < len(args) and not args[index + 1].startswith("--"):
            options[_option_key(part)] = args[index + 1]
            index += 2
            continue
        if part.startswith("--"):
            flags.append(_option_key(part))
            index += 1
            continue
        remainder.append(part)
        index += 1
    return ParsedSlash(options=options, flags=tuple(flags), remainder=tuple(remainder))


def _option_key(value: str) -> str:
    return value[2:].replace("-", "_")
