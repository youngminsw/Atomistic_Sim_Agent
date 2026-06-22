from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    auto_compact_agent_session,
    compact_agent_session,
    replay_agent_compaction,
)
from sim_agent.cli.tui_state import initial_state


def test_auto_compaction_requires_manual_replay_gate(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "md_agent", "user", "first")

    result = auto_compact_agent_session(
        state.session_dir,
        "md_agent",
        AutoCompactionPolicy(new_message_threshold=1),
    )

    errors = _jsonl(state.session_dir / "agent_sessions" / "md_agent" / "compact_errors.jsonl")
    assert result.status == "blocked"
    assert result.blocker == "manual_replay_required"
    assert errors[-1]["blocker"] == "manual_replay_required"


def test_auto_compaction_runs_after_manual_replay_and_new_messages(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "qa_agent", "user", "audit seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="qa_agent", compact_id="manual-qa-001", summary="manual qa summary"),
    )
    replayed = replay_agent_compaction(state.session_dir, "qa_agent")
    append_agent_message(state.session_dir, "qa_agent", "assistant", "audit response")
    append_agent_message(state.session_dir, "qa_agent", "user", "follow-up")

    result = auto_compact_agent_session(
        state.session_dir,
        "qa_agent",
        AutoCompactionPolicy(new_message_threshold=2),
    )

    summary = json.loads((state.session_dir / "agent_sessions" / "qa_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    compactions = _jsonl(state.session_dir / "agent_sessions" / "qa_agent" / "compactions.jsonl")
    assert replayed.status == "succeeded"
    assert result.status == "succeeded"
    assert result.compact_status == "auto_compacted"
    assert summary["compact_mode"] == "auto"
    assert summary["compact_id"] == "auto-qa_agent-3"
    assert summary["message_count"] == 3
    assert compactions[-1]["compact_id"] == "auto-qa_agent-3"


def test_auto_compaction_blocks_when_manual_summary_was_not_replayed(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "ml_mdn_agent", "user", "seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="ml_mdn_agent", compact_id="manual-mdn-001", summary="manual mdn summary"),
    )
    append_agent_message(state.session_dir, "ml_mdn_agent", "assistant", "candidate")

    result = auto_compact_agent_session(
        state.session_dir,
        "ml_mdn_agent",
        AutoCompactionPolicy(new_message_threshold=1),
    )

    errors = _jsonl(state.session_dir / "agent_sessions" / "ml_mdn_agent" / "compact_errors.jsonl")
    summary = json.loads((state.session_dir / "agent_sessions" / "ml_mdn_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    assert result.status == "blocked"
    assert result.blocker == "manual_replay_required"
    assert errors[-1]["blocker"] == "manual_replay_required"
    assert summary["compact_id"] == "manual-mdn-001"


def test_agent_message_append_auto_compacts_after_default_threshold(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "research_graphdb_agent", "user", "seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="research_graphdb_agent", compact_id="manual-rg-001", summary="source seed"),
    )
    replay_agent_compaction(state.session_dir, "research_graphdb_agent")

    for index in range(AutoCompactionPolicy().new_message_threshold):
        append_agent_message(state.session_dir, "research_graphdb_agent", "user", f"source update {index}")

    summary = json.loads(
        (state.session_dir / "agent_sessions" / "research_graphdb_agent" / "compact_summary.json").read_text(encoding="utf-8")
    )
    assert summary["compact_mode"] == "auto"
    assert summary["compact_id"] == "auto-research_graphdb_agent-33"
    assert summary["message_count"] == 33


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
