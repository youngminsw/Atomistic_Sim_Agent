from __future__ import annotations

from pathlib import Path
from typing import Final


PATH_TRIM_MARKER: Final = "…"


def display_path(path: Path | None, *, keep_parts: int = 3) -> str:
    if path is None:
        return ""
    parts = path.parts
    if len(parts) <= keep_parts + 1:
        return str(path)
    return str(Path(PATH_TRIM_MARKER, *parts[-keep_parts:]))
