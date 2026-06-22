from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never


WorkerAdapterKind = Literal["disabled", "tmux", "process"]


@dataclass(frozen=True, slots=True)
class WorkerAdapterConfig:
    kind: WorkerAdapterKind = "disabled"
    command: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkerAdapterStatus:
    kind: WorkerAdapterKind
    status: str
    enabled: bool
    default_runtime_attached: bool
    blocker: str | None = None


def default_worker_adapter_config() -> WorkerAdapterConfig:
    return WorkerAdapterConfig()


def worker_adapter_status(config: WorkerAdapterConfig) -> WorkerAdapterStatus:
    match config.kind:
        case "disabled":
            return WorkerAdapterStatus("disabled", "disabled", False, False, "disabled")
        case "tmux" | "process":
            if not config.command:
                return WorkerAdapterStatus(config.kind, "blocked", False, False, "missing_worker_command")
            return WorkerAdapterStatus(config.kind, "ready", True, False)
        case unreachable:
            assert_never(unreachable)
