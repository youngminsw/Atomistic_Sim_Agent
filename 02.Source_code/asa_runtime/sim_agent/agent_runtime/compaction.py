from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from sim_agent.schemas._parse import JsonMap

from .agent_registry import load_agent_registry
from .agent_session_io import append_agent_event
from .compaction_policy import (
    COMPACT_LEDGER_NAME,
    COMPACT_SCHEMA_VERSION,
    COMPACT_SUMMARY_NAME,
    MIN_RETAINED_MESSAGES,
    MIN_RETAINED_TURNS,
    RETAINED_TAIL_POLICY,
    CompactionMode,
    CompactionPreparation,
    CompactionSummarySource,
    LedgerCounts,
    ProviderContextCompactionBlocked,
    activation_blocker,
    int_value,
    ledger_counts,
    prepare_compaction,
    replay_blocker,
    str_value,
    summary_poison_blocker,
)
from .compaction_semantic import (
    CompactionSemanticSummarizer,
    SemanticSummaryUnavailable,
    build_semantic_summary_request,
    extract_semantic_file_operations,
    semantic_prompt_contract,
    upsert_file_operations,
)
from .compaction_store import append_jsonl, atomic_write_json, read_json, read_jsonl
from sim_agent.compaction_tokens import (
    DEFAULT_AUTO_COMPACT_THRESHOLD_PERCENT,
    DEFAULT_AUTO_COMPACT_THRESHOLD_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_RESERVE_TOKENS,
    CompactionTokenBudget,
    CompactionTokenSettings,
    compaction_budget_for_model,
    estimate_messages_tokens,
    estimate_text_tokens,
    should_compact_tokens,
)
from .global_session_store import backup_legacy_json
from .provider_context_projection import provider_visible_agent_context


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    agent_id: str
    compact_id: str
    summary: str = ""
    compact_mode: CompactionMode = "manual"
    summary_source: CompactionSummarySource = "manual_generated"
    additional_focus: str = ""


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
    provider: str = "openai-codex"
    model: str = "gpt-5-codex"
    enabled: bool = True
    threshold_percent: int = DEFAULT_AUTO_COMPACT_THRESHOLD_PERCENT
    threshold_tokens: int = DEFAULT_AUTO_COMPACT_THRESHOLD_TOKENS
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS

    def token_settings(self) -> CompactionTokenSettings:
        return CompactionTokenSettings(
            enabled=self.enabled,
            threshold_percent=self.threshold_percent,
            threshold_tokens=self.threshold_tokens,
            reserve_tokens=self.reserve_tokens,
            keep_recent_tokens=self.keep_recent_tokens,
            context_window_tokens=self.context_window_tokens,
        )

    def token_budget(self) -> CompactionTokenBudget | None:
        return compaction_budget_for_model(self.provider, self.model, self.token_settings())


@dataclass(frozen=True, slots=True)
class AutoCompactionResult:
    status: str
    compact_status: str
    agent_id: str
    compact_id: str
    raw_message_count: int
    estimated_context_tokens: int
    threshold_tokens: int
    context_window_tokens: int
    summary_path: Path
    blocker: str | None = None


def compact_agent_session(
    session_dir: Path,
    request: CompactionRequest,
    *,
    summarizer: CompactionSemanticSummarizer | None = None,
    policy: AutoCompactionPolicy = AutoCompactionPolicy(),
) -> CompactionResult:
    handle = load_agent_registry(session_dir).handles[request.agent_id]
    counts = ledger_counts(handle.messages_path, handle.events_path)
    if counts is None:
        return _blocked(handle.session_dir, request, "corrupt_ledger")
    messages = read_jsonl(handle.messages_path)
    events = read_jsonl(handle.events_path)
    if messages is None or events is None:
        return _blocked(handle.session_dir, request, "corrupt_ledger")
    preparation = prepare_compaction(tuple(messages), tuple(events), policy.keep_recent_tokens)
    if preparation is None:
        return _blocked(handle.session_dir, request, "unsafe_compaction_boundary")
    summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
    previous_payload = read_json(summary_path)
    try:
        semantic = _semantic_summary_payload(request, tuple(messages), preparation, previous_payload, summarizer)
    except SemanticSummaryUnavailable as exc:
        return _blocked(handle.session_dir, request, str(exc))
    if semantic is None:
        return _blocked(handle.session_dir, request, "semantic_summarizer_unavailable")
    payload = _summary_payload(request, handle.agent_session_id, counts, preparation, semantic)
    backup_legacy_json(summary_path, COMPACT_SCHEMA_VERSION)
    atomic_write_json(summary_path, payload)
    append_jsonl(handle.session_dir / COMPACT_LEDGER_NAME, payload)
    append_agent_event(session_dir, request.agent_id, _completed_event_type(request.compact_mode), request.compact_id)
    return CompactionResult("succeeded", "compacted", request.agent_id, request.compact_id, handle.session_dir / COMPACT_SUMMARY_NAME)


def replay_agent_compaction(session_dir: Path, agent_id: str) -> CompactionResult:
    handle = load_agent_registry(session_dir).handles[agent_id]
    summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
    payload = read_json(summary_path)
    if payload is None:
        return _blocked(handle.session_dir, CompactionRequest(agent_id, "", ""), "compact_summary_missing")
    if payload == {}:
        return _blocked(handle.session_dir, CompactionRequest(agent_id, "", ""), "corrupt_summary")
    request = CompactionRequest(agent_id, str_value(payload, "compact_id"), str_value(payload, "summary"))
    poison_blocker = summary_poison_blocker(payload)
    if poison_blocker is not None:
        return _blocked(handle.session_dir, request, poison_blocker)
    counts = ledger_counts(handle.messages_path, handle.events_path)
    if counts is None:
        return _blocked(handle.session_dir, request, "corrupt_ledger")
    blocker = replay_blocker(handle.messages_path, handle.events_path, counts, payload)
    if blocker is not None:
        return _blocked(handle.session_dir, request, blocker)
    if str_value(payload, "schema_version") != COMPACT_SCHEMA_VERSION:
        messages = read_jsonl(handle.messages_path)
        events = read_jsonl(handle.events_path)
        if messages is None or events is None:
            return _blocked(handle.session_dir, request, "corrupt_ledger")
        preparation = prepare_compaction(tuple(messages), tuple(events), DEFAULT_KEEP_RECENT_TOKENS)
        if preparation is None:
            return _blocked(handle.session_dir, request, "unsafe_compaction_boundary")
        semantic = _legacy_replay_summary_payload(request.summary, payload)
        payload = _summary_payload(request, handle.agent_session_id, counts, preparation, semantic)
    _mark_replay_passed(summary_path, payload, counts)
    return CompactionResult("succeeded", "replayed", agent_id, request.compact_id, summary_path)


def auto_compact_agent_session(
    session_dir: Path,
    agent_id: str,
    policy: AutoCompactionPolicy = AutoCompactionPolicy(),
    *,
    summarizer: CompactionSemanticSummarizer | None = None,
) -> AutoCompactionResult:
    handle = load_agent_registry(session_dir).handles[agent_id]
    counts = ledger_counts(handle.messages_path, handle.events_path)
    compact_id = f"auto-{agent_id}-{counts.message_count if counts is not None else 0}"
    if counts is None:
        blocked = _blocked(handle.session_dir, _auto_request(agent_id, compact_id), "corrupt_ledger")
        return _auto_result(blocked, 0, 0, 0, 0)
    budget = policy.token_budget()
    if budget is None:
        blocked = _blocked(handle.session_dir, _auto_request(agent_id, compact_id, counts), "unknown_model_context_window")
        return _auto_result(blocked, counts.message_count, 0, 0, 0)
    if not budget.enabled:
        return AutoCompactionResult(
            "skipped",
            "disabled",
            agent_id,
            compact_id,
            counts.message_count,
            0,
            0,
            0,
            summary_path=handle.session_dir / COMPACT_SUMMARY_NAME,
        )
    summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
    summary = read_json(summary_path)
    if summary is None:
        messages = read_jsonl(handle.messages_path)
        estimated_context_tokens = estimate_messages_tokens(tuple(messages or ()))
        if not should_compact_tokens(estimated_context_tokens, budget):
            return AutoCompactionResult(
                "skipped",
                "below_token_threshold",
                agent_id,
                compact_id,
                counts.message_count,
                estimated_context_tokens,
                budget.threshold_tokens,
                budget.context_window_tokens,
                summary_path,
            )
        request = _auto_request(agent_id, compact_id, counts)
        compacted = compact_agent_session(session_dir, request, summarizer=summarizer, policy=policy)
        return _auto_result(
            compacted,
            counts.message_count,
            estimated_context_tokens,
            budget.threshold_tokens,
            budget.context_window_tokens,
            compact_status="auto_compacted",
        )
    if summary == {}:
        blocked = _blocked(handle.session_dir, _auto_request(agent_id, compact_id), "corrupt_summary")
        return _auto_result(blocked, counts.message_count, 0, budget.threshold_tokens, budget.context_window_tokens)
    existing_blocker = activation_blocker(summary, handle.messages_path, handle.events_path)
    if existing_blocker is not None:
        blocked = _blocked(handle.session_dir, _auto_request(agent_id, compact_id), existing_blocker)
        return _auto_result(blocked, counts.message_count, 0, budget.threshold_tokens, budget.context_window_tokens)
    try:
        visible_context = provider_visible_agent_context(handle)
    except ProviderContextCompactionBlocked as exc:
        blocked = _blocked(handle.session_dir, _auto_request(agent_id, compact_id), str(exc))
        return _auto_result(blocked, counts.message_count, 0, budget.threshold_tokens, budget.context_window_tokens)
    estimated_context_tokens = estimate_text_tokens(visible_context.compact_summary) + estimate_messages_tokens(
        visible_context.messages,
    )
    if not should_compact_tokens(estimated_context_tokens, budget):
        return AutoCompactionResult(
            "skipped",
            "below_token_threshold",
            agent_id,
            compact_id,
            counts.message_count,
            estimated_context_tokens,
            budget.threshold_tokens,
            budget.context_window_tokens,
            summary_path,
        )
    request = _auto_request(agent_id, compact_id, counts)
    compacted = compact_agent_session(session_dir, request, summarizer=summarizer, policy=policy)
    return _auto_result(
        compacted,
        counts.message_count,
        estimated_context_tokens,
        budget.threshold_tokens,
        budget.context_window_tokens,
        compact_status="auto_compacted",
    )


def _summary_payload(
    request: CompactionRequest,
    agent_session_id: str,
    counts: LedgerCounts,
    preparation: CompactionPreparation,
    semantic: JsonMap,
) -> JsonMap:
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "compact_id": request.compact_id,
        "agent_id": request.agent_id,
        "agent_session_id": agent_session_id,
        "compact_mode": request.compact_mode,
        "summary_source": str_value(semantic, "summary_source") or request.summary_source,
        "manual_replay_status": "passed" if request.compact_mode == "auto" else "required",
        "summary": str_value(semantic, "summary"),
        "short_summary": str_value(semantic, "short_summary"),
        "semantic_prompt_contract": semantic.get("semantic_prompt_contract", {}),
        "semantic_details": semantic.get("semantic_details", {}),
        "preserve_data": semantic.get("preserve_data", {}),
        "provider_cache_invalidated": True,
        "provider_session_reset": True,
        "message_count": counts.message_count,
        "event_count": counts.event_count,
        "raw_message_count": preparation.raw_message_count,
        "recent_message_count": preparation.recent_message_count,
        "compacted_message_count": preparation.compacted_message_count,
        "raw_event_count": preparation.raw_event_count,
        "compacted_event_count": preparation.compacted_event_count,
        "first_kept_message_sequence": preparation.first_kept_message_sequence,
        "summary_cutoff_message_sequence": preparation.summary_cutoff_message_sequence,
        "first_kept_event_sequence": preparation.first_kept_event_sequence,
        "summary_cutoff_event_sequence": preparation.summary_cutoff_event_sequence,
        "last_message_sequence": counts.last_message_sequence,
        "last_event_sequence": counts.last_event_sequence,
        "retained_tail_policy": RETAINED_TAIL_POLICY,
        "min_retained_messages": MIN_RETAINED_MESSAGES,
        "min_retained_turns": MIN_RETAINED_TURNS,
        "keep_recent_tokens": preparation.keep_recent_tokens,
        "retained_tail_token_estimate": preparation.retained_tail_token_estimate,
        "compacted_token_estimate": preparation.compacted_token_estimate,
        "turn_boundary_preserved": preparation.turn_boundary_preserved,
        "created_at": time.time(),
    }


def _semantic_summary_payload(
    request: CompactionRequest,
    messages: tuple[JsonMap, ...],
    preparation: CompactionPreparation,
    previous_payload: JsonMap | None,
    summarizer: CompactionSemanticSummarizer | None,
) -> JsonMap | None:
    if summarizer is None:
        return None
    previous_summary = str_value(previous_payload, "summary") if previous_payload else ""
    summary_request = build_semantic_summary_request(
        agent_id=request.agent_id,
        compact_id=request.compact_id,
        compact_mode=request.compact_mode,
        summary_source=request.summary_source,
        messages=messages,
        first_kept_sequence=preparation.first_kept_message_sequence,
        summary_cutoff_sequence=preparation.summary_cutoff_message_sequence,
        previous_summary=previous_summary,
        additional_focus=request.additional_focus or request.summary,
    )
    result = summarizer.summarize(summary_request)
    semantic_details = extract_semantic_file_operations(messages, _semantic_details(previous_payload))
    read_files = tuple(_string_sequence(semantic_details.get("readFiles")))
    modified_files = tuple(_string_sequence(semantic_details.get("modifiedFiles")))
    summary = upsert_file_operations(result.summary, read_files, modified_files)
    preserve_data = dict(result.preserve_data) if result.preserve_data is not None else _preserve_data(previous_payload)
    return {
        "summary": summary,
        "short_summary": result.short_summary,
        "summary_source": "llm_semantic",
        "semantic_prompt_contract": semantic_prompt_contract(),
        "semantic_details": semantic_details,
        "preserve_data": preserve_data,
    }


def _legacy_replay_summary_payload(summary: str, previous_payload: JsonMap | None) -> JsonMap:
    return {
        "summary": summary,
        "short_summary": str_value(previous_payload, "short_summary") if previous_payload else "",
        "summary_source": "manual_generated",
        "semantic_prompt_contract": semantic_prompt_contract(),
        "semantic_details": _semantic_details(previous_payload),
        "preserve_data": _preserve_data(previous_payload),
    }


def _semantic_details(payload: JsonMap | None) -> JsonMap:
    if payload is None:
        return {}
    value = payload.get("semantic_details")
    return value if isinstance(value, Mapping) else {}


def _preserve_data(payload: JsonMap | None) -> JsonMap:
    if payload is None:
        return {}
    value = payload.get("preserve_data")
    return dict(value) if isinstance(value, Mapping) else {}


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _blocked(agent_dir: Path, request: CompactionRequest, blocker: str) -> CompactionResult:
    payload: JsonMap = {
        "schema_version": "asa_agent_compact_error_v2",
        "at": time.time(),
        "agent_id": request.agent_id,
        "compact_id": request.compact_id,
        "status": "blocked",
        "blocker": blocker,
    }
    append_jsonl(agent_dir / "compact_errors.jsonl", payload)
    return CompactionResult("blocked", "blocked", request.agent_id, request.compact_id, agent_dir / COMPACT_SUMMARY_NAME, blocker)


def _auto_result(
    result: CompactionResult,
    raw_message_count: int,
    estimated_context_tokens: int,
    threshold_tokens: int,
    context_window_tokens: int,
    *,
    compact_status: str | None = None,
) -> AutoCompactionResult:
    return AutoCompactionResult(
        result.status,
        compact_status or result.compact_status,
        result.agent_id,
        result.compact_id,
        raw_message_count,
        estimated_context_tokens,
        threshold_tokens,
        context_window_tokens,
        result.summary_path,
        result.blocker,
    )


def _auto_request(agent_id: str, compact_id: str, counts: LedgerCounts | None = None) -> CompactionRequest:
    return CompactionRequest(agent_id, compact_id, "", compact_mode="auto", summary_source="auto_generated")


def _mark_replay_passed(path: Path, payload: JsonMap, counts: LedgerCounts) -> None:
    replayed = dict(payload)
    replayed.update(
        {
            "schema_version": COMPACT_SCHEMA_VERSION,
            "manual_replay_status": "passed",
            "replayed_at": time.time(),
            "replay_message_count": counts.message_count,
            "replay_event_count": counts.event_count,
        }
    )
    backup_legacy_json(path, COMPACT_SCHEMA_VERSION)
    atomic_write_json(path, replayed)


def _completed_event_type(mode: CompactionMode) -> str:
    match mode:
        case "manual":
            return "manual_compaction_completed"
        case "auto":
            return "auto_compaction_completed"
        case unreachable:
            assert_never(unreachable)
