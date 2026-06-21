from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

from sim_agent.schemas._parse import JsonMap

from .agent_registry import load_agent_registry
from .agent_session_io import append_agent_event
from .compaction_store import append_jsonl, atomic_write_json, read_json, read_jsonl


COMPACT_SCHEMA_VERSION: Final = "asa_agent_compact_summary_v1"
COMPACT_SUMMARY_NAME: Final = "compact_summary.json"
COMPACT_LEDGER_NAME: Final = "compactions.jsonl"
COMPACT_ERROR_NAME: Final = "compact_errors.jsonl"
AUTO_COMPACT_NEW_MESSAGE_THRESHOLD: Final = 32
CompactionMode = Literal["manual", "auto"]


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    agent_id: str
    compact_id: str
    summary: str
    compact_mode: CompactionMode = "manual"


@dataclass(frozen=True, slots=True)
class CompactionResult:
    status: str
    compact_status: str
    agent_id: str
    compact_id: str
    summary_path: Path
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class AutoCompactionPolicy:
    new_message_threshold: int = AUTO_COMPACT_NEW_MESSAGE_THRESHOLD


@dataclass(frozen=True, slots=True)
class AutoCompactionResult:
    status: str
    compact_status: str
    agent_id: str
    compact_id: str
    message_count: int
    new_message_count: int
    summary_path: Path
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerCounts:
    message_count: int
    event_count: int
    last_message_sequence: int
    last_event_sequence: int


def compact_agent_session(session_dir: Path, request: CompactionRequest) -> CompactionResult:
    handle = load_agent_registry(session_dir).handles[request.agent_id]
    counts = _ledger_counts(handle.messages_path, handle.events_path)
    if counts is None:
        return _blocked(handle.session_dir, request, "corrupt_ledger")
    payload = _summary_payload(request, handle.agent_session_id, counts)
    atomic_write_json(handle.session_dir / COMPACT_SUMMARY_NAME, payload)
    append_jsonl(handle.session_dir / COMPACT_LEDGER_NAME, payload)
    append_agent_event(session_dir, request.agent_id, _completed_event_type(request.compact_mode), request.compact_id)
    return CompactionResult("succeeded", "compacted", request.agent_id, request.compact_id, handle.session_dir / COMPACT_SUMMARY_NAME)


def replay_agent_compaction(session_dir: Path, agent_id: str) -> CompactionResult:
    handle = load_agent_registry(session_dir).handles[agent_id]
    summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
    payload = _read_summary_payload(summary_path)
    if payload is None:
        return _blocked(handle.session_dir, CompactionRequest(agent_id, "", ""), "compact_summary_missing")
    if payload == {}:
        return _blocked(handle.session_dir, CompactionRequest(agent_id, "", ""), "corrupt_summary")
    counts = _ledger_counts(handle.messages_path, handle.events_path)
    request = CompactionRequest(agent_id, _str_value(payload, "compact_id"), _str_value(payload, "summary"))
    if counts is None:
        return _blocked(handle.session_dir, request, "corrupt_ledger")
    if _has_replay_mismatch(handle.messages_path, handle.events_path, counts, payload):
        return _blocked(handle.session_dir, request, "compact_replay_mismatch")
    _mark_replay_passed(summary_path, payload, counts)
    return CompactionResult("succeeded", "replayed", agent_id, request.compact_id, summary_path)


def auto_compact_agent_session(
    session_dir: Path,
    agent_id: str,
    policy: AutoCompactionPolicy = AutoCompactionPolicy(),
) -> AutoCompactionResult:
    handle = load_agent_registry(session_dir).handles[agent_id]
    counts = _ledger_counts(handle.messages_path, handle.events_path)
    compact_id = f"auto-{agent_id}-{counts.message_count if counts is not None else 0}"
    if counts is None:
        blocked = _blocked(
            handle.session_dir,
            CompactionRequest(agent_id=agent_id, compact_id=compact_id, summary="", compact_mode="auto"),
            "corrupt_ledger",
        )
        return _auto_result(blocked, 0, 0)
    summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
    summary = _read_summary_payload(summary_path)
    if summary is None:
        if counts.message_count < policy.new_message_threshold:
            return AutoCompactionResult(
                "skipped",
                "below_threshold",
                agent_id,
                compact_id,
                counts.message_count,
                counts.message_count,
                summary_path,
            )
        blocked = _blocked(
            handle.session_dir,
            CompactionRequest(agent_id=agent_id, compact_id=compact_id, summary="", compact_mode="auto"),
            "manual_replay_required",
        )
        return _auto_result(blocked, counts.message_count, counts.message_count)
    if summary == {}:
        blocked = _blocked(
            handle.session_dir,
            CompactionRequest(agent_id=agent_id, compact_id=compact_id, summary="", compact_mode="auto"),
            "corrupt_summary",
        )
        return _auto_result(blocked, counts.message_count, 0)
    new_message_count = max(0, counts.message_count - _int_value(summary, "message_count"))
    if new_message_count < policy.new_message_threshold:
        return AutoCompactionResult("skipped", "below_threshold", agent_id, compact_id, counts.message_count, new_message_count, summary_path)
    if _str_value(summary, "manual_replay_status") != "passed":
        blocked = _blocked(
            handle.session_dir,
            CompactionRequest(agent_id=agent_id, compact_id=compact_id, summary="", compact_mode="auto"),
            "manual_replay_required",
        )
        return _auto_result(blocked, counts.message_count, new_message_count)
    replayed = replay_agent_compaction(session_dir, agent_id)
    if replayed.status != "succeeded":
        return AutoCompactionResult(
            "blocked",
            "blocked",
            agent_id,
            compact_id,
            counts.message_count,
            new_message_count,
            summary_path,
            replayed.blocker or "manual_replay_required",
        )
    request = CompactionRequest(
        agent_id=agent_id,
        compact_id=compact_id,
        summary=_auto_summary(agent_id, counts, new_message_count),
        compact_mode="auto",
    )
    compacted = compact_agent_session(session_dir, request)
    return _auto_result(compacted, counts.message_count, new_message_count, compact_status="auto_compacted")


def _ledger_counts(messages_path: Path, events_path: Path) -> LedgerCounts | None:
    messages = read_jsonl(messages_path)
    events = read_jsonl(events_path)
    if messages is None or events is None:
        return None
    return LedgerCounts(
        message_count=len(messages),
        event_count=len(events),
        last_message_sequence=_last_sequence(messages),
        last_event_sequence=_last_sequence(events),
    )


def _summary_payload(request: CompactionRequest, agent_session_id: str, counts: LedgerCounts) -> JsonMap:
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "compact_id": request.compact_id,
        "agent_id": request.agent_id,
        "agent_session_id": agent_session_id,
        "compact_mode": request.compact_mode,
        "summary": request.summary,
        "message_count": counts.message_count,
        "event_count": counts.event_count,
        "last_message_sequence": counts.last_message_sequence,
        "last_event_sequence": counts.last_event_sequence,
        "created_at": time.time(),
    }


def _blocked(agent_dir: Path, request: CompactionRequest, blocker: str) -> CompactionResult:
    payload: JsonMap = {
        "at": time.time(),
        "agent_id": request.agent_id,
        "compact_id": request.compact_id,
        "status": "blocked",
        "blocker": blocker,
    }
    append_jsonl(agent_dir / COMPACT_ERROR_NAME, payload)
    return CompactionResult("blocked", "blocked", request.agent_id, request.compact_id, agent_dir / COMPACT_SUMMARY_NAME, blocker)


def _auto_result(
    result: CompactionResult,
    message_count: int,
    new_message_count: int,
    *,
    compact_status: str | None = None,
) -> AutoCompactionResult:
    return AutoCompactionResult(
        result.status,
        compact_status or result.compact_status,
        result.agent_id,
        result.compact_id,
        message_count,
        new_message_count,
        result.summary_path,
        result.blocker,
    )


def _auto_summary(agent_id: str, counts: LedgerCounts, new_message_count: int) -> str:
    return (
        f"Auto compacted {agent_id} after {new_message_count} new messages; "
        f"session now has {counts.message_count} messages and {counts.event_count} events."
    )


def _read_summary_payload(path: Path) -> JsonMap | None:
    return read_json(path)


def _has_replay_mismatch(messages_path: Path, events_path: Path, counts: LedgerCounts, payload: JsonMap) -> bool:
    return (
        counts.message_count < _int_value(payload, "message_count")
        or counts.event_count < _int_value(payload, "event_count")
        or _boundary_sequence_mismatch(messages_path, _int_value(payload, "message_count"), _int_value(payload, "last_message_sequence"))
        or _boundary_sequence_mismatch(events_path, _int_value(payload, "event_count"), _int_value(payload, "last_event_sequence"))
    )


def _boundary_sequence_mismatch(path: Path, record_count: int, expected_sequence: int) -> bool:
    records = read_jsonl(path)
    if records is None or len(records) < record_count:
        return True
    if record_count == 0:
        return expected_sequence != 0
    return _last_sequence(records[:record_count]) != expected_sequence


def _mark_replay_passed(path: Path, payload: JsonMap, counts: LedgerCounts) -> None:
    replayed = dict(payload)
    replayed.update(
        {
            "manual_replay_status": "passed",
            "replayed_at": time.time(),
            "replay_message_count": counts.message_count,
            "replay_event_count": counts.event_count,
        }
    )
    atomic_write_json(path, replayed)


def _completed_event_type(mode: CompactionMode) -> str:
    match mode:
        case "manual":
            return "manual_compaction_completed"
        case "auto":
            return "auto_compaction_completed"
        case unreachable:
            assert_never(unreachable)


def _last_sequence(records: list[JsonMap]) -> int:
    if not records:
        return 0
    value = records[-1].get("sequence")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _str_value(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else ""


def _int_value(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
