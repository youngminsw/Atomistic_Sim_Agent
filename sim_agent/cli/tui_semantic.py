from __future__ import annotations

from collections.abc import Sequence
from io import TextIOBase
from typing import TextIO


class SemanticFilteredTextIO(TextIOBase):
    def __init__(self, inner: TextIO) -> None:
        self._inner = inner

    @property
    def encoding(self) -> str | None:
        return self._inner.encoding

    def fileno(self) -> int:
        return self._inner.fileno()

    def flush(self) -> None:
        self._inner.flush()

    def isatty(self) -> bool:
        return self._inner.isatty()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        visible = "".join(line for line in text.splitlines(keepends=True) if not _is_semantic_line(line))
        if visible:
            self._inner.write(visible)
        return len(text)


def filter_semantic_tty_output(output_stream: TextIO) -> TextIO:
    if output_stream.isatty():
        return SemanticFilteredTextIO(output_stream)
    return output_stream


def semantic_output_enabled(output_stream: TextIO) -> bool:
    return not output_stream.isatty()


def write_semantic_line(output_stream: TextIO, line: str) -> None:
    if semantic_output_enabled(output_stream):
        output_stream.write(f"{line}\n")


def write_semantic_lines(output_stream: TextIO, lines: Sequence[str]) -> None:
    if semantic_output_enabled(output_stream):
        for line in lines:
            output_stream.write(f"{line}\n")


def _is_semantic_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first_token = stripped.split(maxsplit=1)[0]
    if "=" not in first_token:
        return False
    key = first_token.split("=", maxsplit=1)[0]
    return key[:1].islower() and key.replace("_", "").isalnum()
