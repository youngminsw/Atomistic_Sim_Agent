from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from sim_agent.agent_runtime import (
    CompactionRequest,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction import COMPACT_SUMMARY_NAME
from sim_agent.agent_runtime.compaction_policy import COMPACT_SCHEMA_VERSION, activation_blocker
from sim_agent.agent_runtime.compaction_store import read_json, read_jsonl
from sim_agent.agent_runtime.agent_registry import AgentRegistry, AgentSessionHandle
from sim_agent.schemas._parse import JsonMap

from .tui_paths import display_path
from .tui_state import TuiState, append_event

MAX_SUMMARY_SNIPPETS = 5
MAX_SUMMARY_SNIPPET_CHARS = 180


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
    summary = _manual_summary(handle, args[1:])
    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id=handle.agent_id,
            compact_id=compact_id,
            summary=summary,
            summary_source="manual_supplied" if len(args) > 1 else "manual_generated",
        ),
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


def _manual_summary(handle: AgentSessionHandle, extra_tokens: Sequence[str]) -> str:
    if extra_tokens:
        return " ".join(extra_tokens)[:2000]
    messages = read_jsonl(handle.messages_path)
    if messages is None:
        return f"Manual TUI compaction requested for {handle.agent_id}; message ledger is corrupt."
    snippets = tuple(_message_snippet(record) for record in messages[-MAX_SUMMARY_SNIPPETS:])
    visible = tuple(snippet for snippet in snippets if snippet)
    if not visible:
        return f"Manual TUI compaction for {handle.agent_id}; no messages recorded yet."
    return f"Manual TUI compaction for {handle.agent_id}. Latest context: {' | '.join(visible)}"


def _message_snippet(record: JsonMap) -> str:
    role = _field(record, "role", "message")
    content = _field(record, "content", "")
    if not content:
        return ""
    return f"{role}: {_trim(content, MAX_SUMMARY_SNIPPET_CHARS)}"


def _target_handle(agent_id: str, registry: AgentRegistry) -> AgentSessionHandle | None:
    return registry.handles.get(agent_id.removeprefix("@"))


def _field(payload: JsonMap, field: str, fallback: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else fallback


def _trim(value: str, limit: int) -> str:
    cleaned = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."
