from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal


GLOBAL_SESSION_SCHEMA_VERSION: Final = "asa_global_session_v1"
GLOBAL_SESSION_LEDGER_NAME: Final = "global_session.json"
GLOBAL_SESSION_EVENTS_NAME: Final = "global_session_events.jsonl"
GLOBAL_SESSION_INDEX_NAME: Final = "global_session_index.jsonl"
LEGACY_SESSION_LEDGER_NAME: Final = "asa_session.json"
ORCHESTRATOR_AGENT_ID: Final = "orchestrator"
ResumeTarget = Literal["latest"] | str | None


@dataclass(frozen=True, slots=True)
class GlobalSessionModel:
    provider: str
    name: str
    reasoning_effort: str
    base_url: str
    auth_mode: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class GlobalSessionPaths:
    global_session: Path
    global_events: Path
    legacy_session: Path
    agent_sessions: Path
    message_bus: Path


@dataclass(frozen=True, slots=True)
class GlobalSessionRecord:
    schema_version: str
    session_id: str
    session_dir: Path
    created_at: float
    updated_at: float
    last_sequence: int
    model: GlobalSessionModel
    agent_ids: tuple[str, ...]
    paths: GlobalSessionPaths
    source: str


@dataclass(frozen=True, slots=True)
class GlobalSessionOpenRequest:
    requested_dir: Path | None
    default_root: Path
    model: GlobalSessionModel
    resume: ResumeTarget = None


@dataclass(frozen=True, slots=True)
class GlobalSessionOpenResult:
    record: GlobalSessionRecord
    opened_as: Literal["created", "resumed", "migrated"]


@dataclass(frozen=True, slots=True)
class GlobalSessionError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason
