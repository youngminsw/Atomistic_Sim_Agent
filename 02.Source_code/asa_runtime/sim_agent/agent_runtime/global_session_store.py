from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, as_str
from sim_agent.schemas.errors import SchemaValidationError

from .global_session_types import (
    GLOBAL_SESSION_EVENT_SCHEMA_VERSION,
    GLOBAL_SESSION_EVENTS_NAME,
    GLOBAL_SESSION_INDEX_NAME,
    GLOBAL_SESSION_LEDGER_NAME,
    GLOBAL_SESSION_SCHEMA_VERSION,
    LEGACY_SESSION_LEDGER_NAME,
    ORCHESTRATOR_AGENT_ID,
    GlobalSessionModel,
    GlobalSessionPaths,
    GlobalSessionRecord,
)


SCHEMA_BACKUP_SUFFIX = ".pre-v2.bak"


def append_global_session_event(
    session_dir: Path,
    event_type: str,
    summary: str,
    *,
    actor: str = ORCHESTRATOR_AGENT_ID,
) -> None:
    ledger_path = session_dir / GLOBAL_SESSION_LEDGER_NAME
    if not ledger_path.is_file():
        return
    record = load_global_session(session_dir)
    at = time.time()
    sequence = record.last_sequence + 1
    payload: JsonMap = {
        "schema_version": GLOBAL_SESSION_EVENT_SCHEMA_VERSION,
        "at": at,
        "sequence": sequence,
        "session_id": record.session_id,
        "event_type": event_type,
        "actor": actor,
        "summary": summary,
    }
    record.paths.global_events.parent.mkdir(parents=True, exist_ok=True)
    with record.paths.global_events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    write_global_session(replace(record, updated_at=at, last_sequence=sequence))


def load_global_session(session_dir: Path) -> GlobalSessionRecord:
    payload = as_mapping(json.loads((session_dir / GLOBAL_SESSION_LEDGER_NAME).read_text(encoding="utf-8")), "global_session")
    model_payload = as_mapping(payload.get("model"), "global_session.model")
    paths_payload = payload.get("paths")
    return GlobalSessionRecord(
        schema_version=GLOBAL_SESSION_SCHEMA_VERSION,
        session_id=as_str(payload.get("session_id"), "global_session.session_id"),
        session_dir=Path(as_str(payload.get("session_dir"), "global_session.session_dir")),
        created_at=_as_float(payload.get("created_at"), "global_session.created_at"),
        updated_at=_as_float(payload.get("updated_at"), "global_session.updated_at"),
        last_sequence=_as_int(payload.get("last_sequence"), "global_session.last_sequence"),
        model=_model_from_payload(model_payload),
        agent_ids=tuple(_as_str_list(payload.get("agent_ids"), "global_session.agent_ids")),
        paths=_paths_from_payload(paths_payload, session_dir),
        source=as_str(payload.get("source"), "global_session.source"),
    )


def write_global_session(record: GlobalSessionRecord) -> None:
    payload: JsonMap = {
        "schema_version": record.schema_version,
        "session_id": record.session_id,
        "session_dir": str(record.session_dir),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_sequence": record.last_sequence,
        "model": asdict(record.model),
        "agent_ids": list(record.agent_ids),
        "paths": {
            "global_session": str(record.paths.global_session),
            "global_events": str(record.paths.global_events),
            "legacy_session": str(record.paths.legacy_session),
            "agent_sessions": str(record.paths.agent_sessions),
            "message_bus": str(record.paths.message_bus),
        },
        "source": record.source,
    }
    backup_legacy_json(record.paths.global_session, record.schema_version)
    _atomic_write_json(record.paths.global_session, payload)


def create_session_dirs(record: GlobalSessionRecord) -> None:
    record.session_dir.mkdir(parents=True, exist_ok=True)
    record.paths.agent_sessions.mkdir(parents=True, exist_ok=True)
    record.paths.message_bus.mkdir(parents=True, exist_ok=True)


def paths_for(session_dir: Path) -> GlobalSessionPaths:
    return GlobalSessionPaths(
        global_session=session_dir / GLOBAL_SESSION_LEDGER_NAME,
        global_events=session_dir / GLOBAL_SESSION_EVENTS_NAME,
        legacy_session=session_dir / LEGACY_SESSION_LEDGER_NAME,
        agent_sessions=session_dir / "agent_sessions",
        message_bus=session_dir / "message_bus",
    )


def backup_legacy_json(path: Path, current_schema_version: str) -> Path | None:
    if not path.is_file():
        return None
    backup_path = path.with_name(f"{path.name}{SCHEMA_BACKUP_SUFFIX}")
    if backup_path.is_file():
        return backup_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        path.replace(backup_path)
        return backup_path
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if schema_version == current_schema_version:
        return None
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def append_index(default_root: Path, record: GlobalSessionRecord) -> None:
    default_root.mkdir(parents=True, exist_ok=True)
    payload: JsonMap = {
        "at": record.created_at,
        "session_id": record.session_id,
        "session_dir": str(record.session_dir),
    }
    with (default_root / GLOBAL_SESSION_INDEX_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def lookup_index(default_root: Path, session_id: str) -> Path | None:
    for entry in reversed(_index_entries(default_root)):
        if entry[0] == session_id:
            return entry[1]
    return None


def latest_index_entry(default_root: Path) -> Path | None:
    entries = _index_entries(default_root)
    if not entries:
        return None
    return entries[-1][1]


def _atomic_write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _index_entries(default_root: Path) -> list[tuple[str, Path]]:
    path = default_root / GLOBAL_SESSION_INDEX_NAME
    if not path.is_file():
        return []
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = as_mapping(json.loads(line), "global_session_index")
        session_dir = Path(as_str(payload.get("session_dir"), "global_session_index.session_dir"))
        if not _is_relative_to(session_dir, default_root):
            continue
        entries.append((as_str(payload.get("session_id"), "global_session_index.session_id"), session_dir))
    return entries


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path_resolved = path.expanduser().resolve()
        root_resolved = root.expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def _model_from_payload(payload: JsonMap) -> GlobalSessionModel:
    return GlobalSessionModel(
        provider=as_str(payload.get("provider"), "global_session.model.provider"),
        name=as_str(payload.get("name"), "global_session.model.name"),
        reasoning_effort=as_str(payload.get("reasoning_effort"), "global_session.model.reasoning_effort"),
        base_url=as_str(payload.get("base_url"), "global_session.model.base_url"),
        auth_mode=as_str(payload.get("auth_mode"), "global_session.model.auth_mode"),
        api_key_env=as_str(payload.get("api_key_env"), "global_session.model.api_key_env"),
    )


def _paths_from_payload(value: object, session_dir: Path) -> GlobalSessionPaths:
    if not isinstance(value, dict):
        return paths_for(session_dir)
    payload = as_mapping(value, "global_session.paths")
    return GlobalSessionPaths(
        global_session=Path(as_str(payload.get("global_session"), "global_session.paths.global_session")),
        global_events=Path(as_str(payload.get("global_events"), "global_session.paths.global_events")),
        legacy_session=Path(as_str(payload.get("legacy_session"), "global_session.paths.legacy_session")),
        agent_sessions=Path(as_str(payload.get("agent_sessions"), "global_session.paths.agent_sessions")),
        message_bus=Path(as_str(payload.get("message_bus"), "global_session.paths.message_bus")),
    )


def _as_float(value: object, field: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise SchemaValidationError(f"{field} must be a number")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")


def _as_str_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{field} must be a list")
    return [as_str(item, f"{field}[]") for item in value]
