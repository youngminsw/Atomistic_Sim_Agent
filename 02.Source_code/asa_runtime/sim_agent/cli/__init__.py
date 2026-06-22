from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import main

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .main import main

        return main
    raise AttributeError(name)
