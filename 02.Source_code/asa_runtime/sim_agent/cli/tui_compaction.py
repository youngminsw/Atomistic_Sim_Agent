from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction import COMPACT_SUMMARY_NAME
from sim_agent.agent_runtime.compaction_policy import COMPACT_SCHEMA_VERSION, activation_blocker
from sim_agent.agent_runtime.compaction_provider_summarizer import ProviderSemanticSummarizer
from sim_agent.agent_runtime.compaction_store import read_json, read_jsonl
from sim_agent.agent_runtime.agent_registry import AgentRegistry, AgentSessionHandle
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import JsonMap

from .tui_paths import display_path
from .tui_state import TuiState, append_event

MAX_COMPACT_FOCUS_CHARS = 2000


@dataclass(frozen=True, slots=True)
class CompactStatusRow:
    agent_id: str
    summary_status: str
    message_count: int
    replay_status: str
    compact_id: str


def handle_compact(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    registry = load_agent_registry(state.session_dir)
    if not args or args[0] == "status":
        _write_compaction_status(registry, output_stream)
        return state
    if args[0] == "replay":
        return _handle_replay(args[1:], state, registry, output_stream)
    return _handle_manual_compaction(args, state, registry, output_stream)


def _handle_manual_compaction(
    args: Sequence[str],
    state: TuiState,
    registry: AgentRegistry,
    output_stream: TextIO,
) -> TuiState:
    handle = _target_handle(args[0], registry)
    if handle is None:
        output_stream.write(f"compact_error=unknown_agent agent={args[0]}\n")
        return state
    if len(args) > 1 and args[1] == "replay":
        return _run_replay(handle.agent_id, state, output_stream)
    compact_id = f"manual-{handle.agent_id}-{int(time.time())}"
    additional_focus = _manual_focus(args[1:])
    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id=handle.agent_id,
            compact_id=compact_id,
            summary_source="manual_generated",
            additional_focus=additional_focus,
        ),
        summarizer=_summarizer_for_state(state),
        policy=_policy_for_state(state),
    )
    output_stream.write("manual_compaction=true\n")
    output_stream.write(f"compact_agent={handle.agent_id}\n")
    output_stream.write(f"compact_status={compacted.compact_status}\n")
    if compacted.blocker is not None:
        output_stream.write(f"compact_blocker={compacted.blocker}\n")
        append_event(state, "manual_compaction_blocked", f"{handle.agent_id}:{compacted.blocker}")
        return state
    replayed = replay_agent_compaction(state.session_dir, handle.agent_id)
    output_stream.write(f"compact_replay_status={replayed.compact_status}\n")
    if replayed.blocker is not None:
        output_stream.write(f"compact_replay_blocker={replayed.blocker}\n")
        append_event(state, "manual_compaction_replay_blocked", f"{handle.agent_id}:{replayed.blocker}")
        return state
    output_stream.write(f"compact_summary_path={display_path(replayed.summary_path)}\n")
    append_event(state, "manual_compaction_replayed", f"{handle.agent_id}:{compact_id}")
    return state


def _handle_replay(
    args: Sequence[str],
    state: TuiState,
    registry: AgentRegistry,
    output_stream: TextIO,
) -> TuiState:
    if not args:
        output_stream.write("compact_error=missing_agent\n")
        return state
    handle = _target_handle(args[0], registry)
    if handle is None:
        output_stream.write(f"compact_error=unknown_agent agent={args[0]}\n")
        return state
    return _run_replay(handle.agent_id, state, output_stream)


def _run_replay(agent_id: str, state: TuiState, output_stream: TextIO) -> TuiState:
    replayed = replay_agent_compaction(state.session_dir, agent_id)
    output_stream.write("manual_compaction_replay=true\n")
    output_stream.write(f"compact_agent={agent_id}\n")
    output_stream.write(f"compact_replay_status={replayed.compact_status}\n")
    if replayed.blocker is not None:
        output_stream.write(f"compact_replay_blocker={replayed.blocker}\n")
        append_event(state, "manual_compaction_replay_blocked", f"{agent_id}:{replayed.blocker}")
        return state
    output_stream.write(f"compact_summary_path={display_path(replayed.summary_path)}\n")
    append_event(state, "manual_compaction_replayed", agent_id)
    return state


def _write_compaction_status(registry: AgentRegistry, output_stream: TextIO) -> None:
    output_stream.write("Compaction Status\n")
    output_stream.write("Agent                Summary    Messages  Replay      Compact ID\n")
    output_stream.write("compact_status_view=true\n")
    for handle in registry.handles.values():
        summary_path = handle.session_dir / COMPACT_SUMMARY_NAME
        summary = read_json(summary_path)
        messages = read_jsonl(handle.messages_path)
        message_count = 0 if messages is None else len(messages)
        if summary is None:
            _write_agent_status(
                output_stream,
                CompactStatusRow(handle.agent_id, "missing", message_count, "required", "-"),
            )
            continue
        if summary == {}:
            _write_agent_status(
                output_stream,
                CompactStatusRow(handle.agent_id, "corrupt", message_count, "blocked", "-"),
            )
            continue
        blocker = activation_blocker(summary, handle.messages_path, handle.events_path)
        replay_status = _field(summary, "manual_replay_status", "passed" if blocker is None else "blocked")
        compact_id = _field(summary, "compact_id", "-")
        summary_status = "rewrite_active" if summary.get("schema_version") == COMPACT_SCHEMA_VERSION and blocker is None else "blocked"
        _write_agent_status(
            output_stream,
            CompactStatusRow(handle.agent_id, summary_status, message_count, replay_status, compact_id),
        )


def _write_agent_status(output_stream: TextIO, row: CompactStatusRow) -> None:
    output_stream.write(
        f"{row.agent_id:<20} {row.summary_status:<10} {row.message_count:<9} "
        f"{row.replay_status:<11} {row.compact_id}\n"
    )
    output_stream.write(f"compact_agent={row.agent_id}\n")
    output_stream.write(f"compact_summary_status={row.summary_status}\n")
    output_stream.write(f"compact_message_count={row.message_count} compact_replay={row.replay_status}\n")
    if row.compact_id != "-":
        output_stream.write(f"compact_id={row.compact_id}\n")


def _manual_focus(extra_tokens: Sequence[str]) -> str:
    return " ".join(extra_tokens).strip()[:MAX_COMPACT_FOCUS_CHARS]


def _target_handle(agent_id: str, registry: AgentRegistry) -> AgentSessionHandle | None:
    return registry.handles.get(agent_id.removeprefix("@"))


def _field(payload: JsonMap, field: str, fallback: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else fallback


def _summarizer_for_state(state: TuiState) -> ProviderSemanticSummarizer:
    return ProviderSemanticSummarizer(
        ModelProviderConfig.from_mapping(
            {
                "provider": state.model.provider,
                "model": state.model.name,
                "reasoning_effort": state.model.reasoning_effort,
                "base_url": state.model.base_url,
                "auth_mode": state.model.auth_mode,
                "api_key_env": state.model.api_key_env,
            }
        )
    )


def _policy_for_state(state: TuiState) -> AutoCompactionPolicy:
    compaction = load_runtime_config().compaction
    return AutoCompactionPolicy(
        provider=state.model.provider,
        model=state.model.name,
        enabled=compaction.enabled,
        threshold_percent=compaction.threshold_percent,
        threshold_tokens=compaction.threshold_tokens,
        reserve_tokens=compaction.reserve_tokens,
        keep_recent_tokens=compaction.keep_recent_tokens,
        context_window_tokens=compaction.context_window_tokens,
    )
