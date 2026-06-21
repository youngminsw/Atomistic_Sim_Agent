from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_runtime import (
    CompactionRequest,
    append_agent_message,
    compact_agent_session,
    replay_agent_compaction,
)
from sim_agent.cli.tui_state import initial_state


def test_manual_compaction_writes_summary_and_replays_cursor(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "md_agent", "user", "first MD instruction")
    append_agent_message(state.session_dir, "md_agent", "assistant", "first MD response")

    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="md_agent", compact_id="compact-md-001", summary="MD context summary"),
    )
    replayed = replay_agent_compaction(state.session_dir, "md_agent")

    summary = json.loads((state.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    compactions = _jsonl(state.session_dir / "agent_sessions" / "md_agent" / "compactions.jsonl")
    assert compacted.status == "succeeded"
    assert replayed.status == "succeeded"
    assert summary["schema_version"] == "asa_agent_compact_summary_v1"
    assert summary["compact_id"] == "compact-md-001"
    assert summary["message_count"] == 2
    assert summary["summary"] == "MD context summary"
    assert compactions[-1]["compact_id"] == "compact-md-001"


def test_manual_compaction_blocks_corrupt_agent_ledger(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "qa_agent", "user", "audit")
    (state.session_dir / "agent_sessions" / "qa_agent" / "messages.jsonl").write_text("{broken-json\n", encoding="utf-8")

    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="qa_agent", compact_id="compact-qa-bad", summary="bad"),
    )

    errors = _jsonl(state.session_dir / "agent_sessions" / "qa_agent" / "compact_errors.jsonl")
    assert compacted.status == "blocked"
    assert compacted.blocker == "corrupt_ledger"
    assert errors[-1]["blocker"] == "corrupt_ledger"


def test_compaction_replay_detects_cursor_mismatch(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "research_graphdb_agent", "user", "collect sources")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="research_graphdb_agent", compact_id="compact-rg-001", summary="source context"),
    )
    (state.session_dir / "agent_sessions" / "research_graphdb_agent" / "messages.jsonl").write_text("", encoding="utf-8")

    replayed = replay_agent_compaction(state.session_dir, "research_graphdb_agent")

    errors = _jsonl(state.session_dir / "agent_sessions" / "research_graphdb_agent" / "compact_errors.jsonl")
    assert replayed.status == "blocked"
    assert replayed.blocker == "compact_replay_mismatch"
    assert errors[-1]["blocker"] == "compact_replay_mismatch"


def test_compaction_replay_detects_same_count_sequence_mismatch(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "feature_scale_agent", "user", "first")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="feature_scale_agent", compact_id="compact-fs-001", summary="feature context"),
    )
    messages_path = state.session_dir / "agent_sessions" / "feature_scale_agent" / "messages.jsonl"
    records = _jsonl(messages_path)
    records[0]["sequence"] = 99
    messages_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")

    replayed = replay_agent_compaction(state.session_dir, "feature_scale_agent")

    errors = _jsonl(state.session_dir / "agent_sessions" / "feature_scale_agent" / "compact_errors.jsonl")
    assert replayed.status == "blocked"
    assert replayed.blocker == "compact_replay_mismatch"
    assert errors[-1]["blocker"] == "compact_replay_mismatch"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
