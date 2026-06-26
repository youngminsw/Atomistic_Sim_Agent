from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from sim_agent.schemas._parse import JsonMap

from sim_agent.compaction_tokens import DEFAULT_KEEP_RECENT_TOKENS, estimate_message_tokens
from .compaction_store import read_jsonl

COMPACT_SCHEMA_VERSION: Final = "asa_agent_compact_summary_v4"
LEGACY_COMPACT_SCHEMA_VERSION: Final = "asa_agent_compact_summary_v3"
LEGACY_COMPACT_SCHEMA_VERSIONS: Final = ("asa_agent_compact_summary_v2", "asa_agent_compact_summary_v3")
COMPACT_SUMMARY_NAME: Final = "compact_summary.json"
COMPACT_LEDGER_NAME: Final = "compactions.jsonl"
RETAINED_TAIL_POLICY: Final = "keep_recent_tokens_with_minimum_turns"
MIN_RETAINED_MESSAGES: Final = 8
MIN_RETAINED_TURNS: Final = 4

CompactionMode = Literal["manual", "auto"]
CompactionSummarySource = Literal["manual_generated", "auto_generated", "llm_semantic"]


@dataclass(frozen=True, slots=True)
class LedgerCounts:
    message_count: int
    event_count: int
    last_message_sequence: int
    last_event_sequence: int


@dataclass(frozen=True, slots=True)
class CompactionPreparation:
    first_kept_message_sequence: int
    summary_cutoff_message_sequence: int
    first_kept_event_sequence: int
    summary_cutoff_event_sequence: int
    raw_message_count: int
    recent_message_count: int
    compacted_message_count: int
    raw_event_count: int
    compacted_event_count: int
    last_message_sequence: int
    last_event_sequence: int
    turn_boundary_preserved: bool
    keep_recent_tokens: int
    retained_tail_token_estimate: int
    compacted_token_estimate: int


@dataclass(frozen=True, slots=True)
class ProviderContextCompaction:
    compact_id: str
    compact_mode: str
    summary_source: str
    first_kept_message_sequence: int
    summary_cutoff_message_sequence: int
    first_kept_event_sequence: int
    summary_cutoff_event_sequence: int
    raw_message_count: int
    provider_visible_message_count: int
    recent_message_count: int
    compacted_message_count: int
    short_summary: str
    provider_cache_invalidated: bool
    preserve_data_openai_remote: bool

    def to_json(self) -> JsonMap:
        return {
            "compact_id": self.compact_id,
            "compact_mode": self.compact_mode,
            "summary_source": self.summary_source,
            "first_kept_message_sequence": self.first_kept_message_sequence,
            "summary_cutoff_message_sequence": self.summary_cutoff_message_sequence,
            "first_kept_event_sequence": self.first_kept_event_sequence,
            "summary_cutoff_event_sequence": self.summary_cutoff_event_sequence,
            "raw_message_count": self.raw_message_count,
            "provider_visible_message_count": self.provider_visible_message_count,
            "recent_message_count": self.recent_message_count,
            "compacted_message_count": self.compacted_message_count,
            "short_summary": self.short_summary,
            "provider_cache_invalidated": self.provider_cache_invalidated,
            "provider_session_reset": self.provider_cache_invalidated,
            "preserve_data_openai_remote": self.preserve_data_openai_remote,
            "rewrite_active": True,
        }


@dataclass(frozen=True, slots=True)
class ProviderContextCompactionBlocked(RuntimeError):
    blocker: str

    def __str__(self) -> str:
        return self.blocker


def ledger_counts(messages_path: Path, events_path: Path) -> LedgerCounts | None:
    messages = read_jsonl(messages_path)
    events = read_jsonl(events_path)
    if messages is None or events is None:
        return None
    return LedgerCounts(
        message_count=len(messages),
        event_count=len(events),
        last_message_sequence=last_sequence(messages),
        last_event_sequence=last_sequence(events),
    )


def prepare_compaction(
    messages: tuple[JsonMap, ...],
    events: tuple[JsonMap, ...],
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
) -> CompactionPreparation | None:
    if _has_missing_sequence(messages) or _has_missing_sequence(events):
        return None
    first_index = _token_tail_first_index(messages, keep_recent_tokens)
    boundary_index = _turn_boundary_index(messages, first_index)
    turn_boundary_preserved = True
    if boundary_index < first_index:
        boundary_token_estimate = sum(estimate_message_tokens(message) for message in messages[boundary_index:])
        if boundary_token_estimate <= max(1, keep_recent_tokens):
            first_index = boundary_index
        else:
            turn_boundary_preserved = False
    else:
        first_index = boundary_index
    retained_tail_token_estimate = sum(estimate_message_tokens(message) for message in messages[first_index:])
    compacted_token_estimate = sum(estimate_message_tokens(message) for message in messages[:first_index])
    first_kept_sequence = sequence_at(messages, first_index)
    cutoff_sequence = sequence_at(messages, first_index - 1) if first_index > 0 else 0
    last_event_sequence = last_sequence(events)
    return CompactionPreparation(
        first_kept_message_sequence=first_kept_sequence if messages else 1,
        summary_cutoff_message_sequence=cutoff_sequence,
        first_kept_event_sequence=last_event_sequence + 1,
        summary_cutoff_event_sequence=last_event_sequence,
        raw_message_count=len(messages),
        recent_message_count=max(0, len(messages) - first_index),
        compacted_message_count=first_index,
        raw_event_count=len(events),
        compacted_event_count=len(events),
        last_message_sequence=last_sequence(messages),
        last_event_sequence=last_event_sequence,
        turn_boundary_preserved=turn_boundary_preserved,
        keep_recent_tokens=max(1, keep_recent_tokens),
        retained_tail_token_estimate=retained_tail_token_estimate,
        compacted_token_estimate=compacted_token_estimate,
    )


def replay_blocker(messages_path: Path, events_path: Path, counts: LedgerCounts, payload: JsonMap) -> str | None:
    message_count = int_value(payload, "raw_message_count", int_value(payload, "message_count"))
    event_count = int_value(payload, "raw_event_count", int_value(payload, "event_count"))
    cutoff_count = max(0, message_count - int_value(payload, "recent_message_count"))
    if has_orphan_tool_result(messages_path, events_path, cutoff_count, event_count):
        return "orphan_tool_result"
    if (
        counts.message_count < message_count
        or counts.event_count < event_count
        or boundary_sequence_mismatch(messages_path, message_count, int_value(payload, "last_message_sequence"))
        or boundary_sequence_mismatch(events_path, event_count, int_value(payload, "last_event_sequence"))
    ):
        return "stale_compact_cursor"
    return None


def activation_blocker(payload: JsonMap, messages_path: Path, events_path: Path) -> str | None:
    schema = str_value(payload, "schema_version")
    if schema in LEGACY_COMPACT_SCHEMA_VERSIONS:
        return "legacy_summary"
    if schema != COMPACT_SCHEMA_VERSION:
        return "corrupt_summary"
    poison = summary_poison_blocker(payload)
    if poison is not None:
        return poison
    counts = ledger_counts(messages_path, events_path)
    if counts is None:
        return "corrupt_ledger"
    replay = replay_blocker(messages_path, events_path, counts, payload)
    if replay is not None:
        return replay
    if str_value(payload, "compact_mode") == "manual" and str_value(payload, "manual_replay_status") != "passed":
        return "manual_replay_required"
    return None


def summary_poison_blocker(payload: JsonMap) -> str | None:
    summary = str_value(payload, "summary").lower()
    poison_markers = (
        "ignore previous instructions",
        "ignore all previous",
        "<system",
        "</system",
        '"role":"system"',
        '"role": "system"',
        "role=system",
        "developer message",
        "system prompt override",
        "function_call_output",
    )
    if any(marker in summary for marker in poison_markers):
        return "compact_summary_poisoned"
    return None


def has_orphan_tool_result(messages_path: Path, events_path: Path, message_count: int, event_count: int) -> bool:
    messages = read_jsonl(messages_path)
    events = read_jsonl(events_path)
    if messages is None or events is None:
        return False
    for record in messages[:message_count]:
        if record.get("role") in {"tool", "function"} or record.get("type") in {"tool_result", "function_call_output"}:
            return True
    pending_tool_selections = 0
    for record in events[:event_count]:
        event_type = record.get("event_type")
        if event_type == "model_tool_selected":
            pending_tool_selections += 1
        elif event_type == "tool_result_appended":
            if pending_tool_selections <= 0:
                return True
            pending_tool_selections -= 1
    return False


def boundary_sequence_mismatch(path: Path, record_count: int, expected_sequence: int) -> bool:
    records = read_jsonl(path)
    if records is None or len(records) < record_count:
        return True
    if record_count == 0:
        return expected_sequence != 0
    return last_sequence(tuple(records[:record_count])) != expected_sequence


def last_sequence(records: tuple[JsonMap, ...] | list[JsonMap]) -> int:
    if not records:
        return 0
    return sequence_at(tuple(records), len(records) - 1)


def sequence_at(records: tuple[JsonMap, ...], index: int) -> int:
    if index < 0 or index >= len(records):
        return 0
    value = records[index].get("sequence")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def str_value(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else ""


def int_value(payload: JsonMap, field: str, default: int = 0) -> int:
    value = payload.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _has_missing_sequence(records: tuple[JsonMap, ...]) -> bool:
    return any(sequence_at(records, index) <= 0 for index in range(len(records)))


def _turn_boundary_index(messages: tuple[JsonMap, ...], first_index: int) -> int:
    if first_index <= 0 or first_index >= len(messages):
        return max(0, first_index)
    return first_index - 1 if messages[first_index].get("role") == "assistant" else first_index


def _token_tail_first_index(messages: tuple[JsonMap, ...], keep_recent_tokens: int) -> int:
    if not messages:
        return 0
    token_limit = max(1, keep_recent_tokens)
    token_total = 0
    retained_messages = 0
    first_index = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        next_tokens = estimate_message_tokens(messages[index])
        if retained_messages > 0 and token_total + next_tokens > token_limit:
            break
        token_total += next_tokens
        retained_messages += 1
        first_index = index
        if token_total >= token_limit:
            break
    return first_index
