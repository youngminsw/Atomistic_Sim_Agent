from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap

from .agent_registry import AgentSessionHandle
from .compaction_policy import (
    COMPACT_SUMMARY_NAME,
    ProviderContextCompaction,
    ProviderContextCompactionBlocked,
    activation_blocker,
    int_value,
    str_value,
)
from .compaction_semantic import render_provider_compaction_summary
from .compaction_store import read_json, read_jsonl


@dataclass(frozen=True, slots=True)
class ProviderVisibleAgentContext:
    messages: tuple[JsonMap, ...]
    compact_summary: str
    compaction: ProviderContextCompaction | None
    raw_message_count: int


def provider_visible_agent_context(handle: AgentSessionHandle) -> ProviderVisibleAgentContext:
    records = read_jsonl(handle.messages_path)
    if records is None:
        return ProviderVisibleAgentContext((), "", None, 0)
    summary = read_json(handle.session_dir / COMPACT_SUMMARY_NAME)
    if summary is None:
        return ProviderVisibleAgentContext(_chat_records(records), "", None, len(records))
    if summary == {}:
        raise ProviderContextCompactionBlocked("corrupt_summary")
    blocker = activation_blocker(summary, handle.messages_path, handle.events_path)
    if blocker is not None:
        raise ProviderContextCompactionBlocked(blocker)
    first_kept = int_value(summary, "first_kept_message_sequence")
    visible = tuple(record for record in (_chat_record(raw) for raw in records) if record and _kept(record, first_kept))
    compaction = ProviderContextCompaction(
        compact_id=str_value(summary, "compact_id"),
        compact_mode=str_value(summary, "compact_mode"),
        summary_source=str_value(summary, "summary_source"),
        first_kept_message_sequence=first_kept,
        summary_cutoff_message_sequence=int_value(summary, "summary_cutoff_message_sequence"),
        first_kept_event_sequence=int_value(summary, "first_kept_event_sequence"),
        summary_cutoff_event_sequence=int_value(summary, "summary_cutoff_event_sequence"),
        raw_message_count=len(records),
        provider_visible_message_count=len(visible),
        recent_message_count=int_value(summary, "recent_message_count"),
        compacted_message_count=int_value(summary, "compacted_message_count"),
        short_summary=str_value(summary, "short_summary"),
        provider_cache_invalidated=_bool_value(summary, "provider_cache_invalidated"),
        preserve_data_openai_remote=_preserve_data_openai_remote(summary),
    )
    return ProviderVisibleAgentContext(visible, render_provider_compaction_summary(str_value(summary, "summary")), compaction, len(records))


def bounded_caller_context(handle: AgentSessionHandle) -> str:
    context = provider_visible_agent_context(handle)
    lines = [f"Caller compact summary: {context.compact_summary}"] if context.compact_summary else []
    for message in context.messages:
        role = str_value(message, "role")
        content = str_value(message, "content")
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _chat_records(records: list[JsonMap]) -> tuple[JsonMap, ...]:
    visible: list[JsonMap] = []
    for record in records:
        chat_record = _chat_record(record)
        if chat_record is not None:
            visible.append(chat_record)
    return tuple(visible)


def _chat_record(record: JsonMap) -> JsonMap | None:
    role = record.get("role")
    content = record.get("content")
    if role not in {"user", "assistant", "system"} or not isinstance(content, str):
        return None
    payload = {"role": role, "content": content}
    sequence = record.get("sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool):
        payload["sequence"] = sequence
    return payload


def _kept(record: JsonMap, first_kept_sequence: int) -> bool:
    sequence = record.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ProviderContextCompactionBlocked("missing_compaction_message_sequence")
    return sequence >= first_kept_sequence


def _bool_value(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    return value if isinstance(value, bool) else False


def _preserve_data_openai_remote(payload: JsonMap) -> bool:
    preserve_data = payload.get("preserve_data")
    if not isinstance(preserve_data, dict):
        return False
    return "openaiRemoteCompaction" in preserve_data
