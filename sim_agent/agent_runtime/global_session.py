from __future__ import annotations

import json
import time
from pathlib import Path

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES
from sim_agent.schemas._parse import as_mapping, as_str

from .agent_registry import ensure_agent_registry
from .global_session_store import (
    append_index,
    create_session_dirs,
    latest_index_entry,
    load_global_session,
    lookup_index,
    paths_for,
    write_global_session,
)
from .global_session_types import (
    GLOBAL_SESSION_LEDGER_NAME,
    GLOBAL_SESSION_SCHEMA_VERSION,
    LEGACY_SESSION_LEDGER_NAME,
    ORCHESTRATOR_AGENT_ID,
    GlobalSessionError,
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    GlobalSessionOpenResult,
    GlobalSessionRecord,
)


def open_global_session(request: GlobalSessionOpenRequest) -> GlobalSessionOpenResult:
    if request.resume is None:
        record = _create_global_session(request)
        ensure_agent_registry(record)
        append_index(request.default_root, record)
        return GlobalSessionOpenResult(record, "created")
    record = _resume_global_session(request)
    ensure_agent_registry(record)
    return GlobalSessionOpenResult(record, "resumed" if record.source != "legacy_migration" else "migrated")


def _create_global_session(request: GlobalSessionOpenRequest) -> GlobalSessionRecord:
    session_id = f"asa-{time.time_ns()}"
    session_dir = request.requested_dir or (request.default_root / "sessions" / session_id)
    now = time.time()
    record = GlobalSessionRecord(
        schema_version=GLOBAL_SESSION_SCHEMA_VERSION,
        session_id=session_id,
        session_dir=session_dir,
        created_at=now,
        updated_at=now,
        last_sequence=0,
        model=request.model,
        agent_ids=_agent_ids(),
        paths=paths_for(session_dir),
        source="created",
    )
    create_session_dirs(record)
    write_global_session(record)
    return record


def _resume_global_session(request: GlobalSessionOpenRequest) -> GlobalSessionRecord:
    session_dir = _resolve_resume_dir(request)
    if (session_dir / GLOBAL_SESSION_LEDGER_NAME).is_file():
        return load_global_session(session_dir)
    if (session_dir / LEGACY_SESSION_LEDGER_NAME).is_file():
        return _migrate_legacy_session(session_dir, request.model)
    raise GlobalSessionError(f"global session not found: {session_dir}")


def _resolve_resume_dir(request: GlobalSessionOpenRequest) -> Path:
    if request.requested_dir is not None and request.resume in {"latest", ""}:
        return request.requested_dir
    if isinstance(request.resume, str) and request.resume not in {"latest", ""}:
        target_path = Path(request.resume)
        if target_path.exists():
            return target_path if target_path.is_dir() else target_path.parent
        indexed = lookup_index(request.default_root, request.resume)
        if indexed is not None:
            return indexed
        return request.default_root / "sessions" / request.resume
    latest = latest_index_entry(request.default_root)
    if latest is not None:
        return latest
    if request.requested_dir is not None:
        return request.requested_dir
    raise GlobalSessionError(f"no resumable global session under {request.default_root}")


def _migrate_legacy_session(session_dir: Path, model: GlobalSessionModel) -> GlobalSessionRecord:
    payload = as_mapping(json.loads((session_dir / LEGACY_SESSION_LEDGER_NAME).read_text(encoding="utf-8")), "asa_session")
    session_id = as_str(payload.get("session_id"), "asa_session.session_id")
    now = time.time()
    record = GlobalSessionRecord(
        schema_version=GLOBAL_SESSION_SCHEMA_VERSION,
        session_id=session_id,
        session_dir=session_dir,
        created_at=now,
        updated_at=now,
        last_sequence=0,
        model=model,
        agent_ids=_agent_ids(),
        paths=paths_for(session_dir),
        source="legacy_migration",
    )
    create_session_dirs(record)
    write_global_session(record)
    return record


def _agent_ids() -> tuple[str, ...]:
    return (ORCHESTRATOR_AGENT_ID, *(role.role_id for role in AGENT_ROLES))
