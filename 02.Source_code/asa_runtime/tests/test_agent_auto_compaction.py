from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    auto_compact_agent_session,
    compact_agent_session,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.cli.tui_state import initial_state


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_auto_compaction_activates_generated_summary_without_manual_replay(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "md_agent", "user", "first")
    summarizer = RecordingSummarizer(SemanticSummaryResult(summary="## Goal\n- auto MD summary"), [])

    result = auto_compact_agent_session(
        state.session_dir,
        "md_agent",
        AutoCompactionPolicy(new_message_threshold=1),
        summarizer=summarizer,
    )

    summary = json.loads((state.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    assert result.status == "succeeded"
    assert result.compact_status == "auto_compacted"
    assert summary["schema_version"] == "asa_agent_compact_summary_v4"
    assert summary["summary_source"] == "llm_semantic"
    assert summary["manual_replay_status"] == "passed"
    assert summarizer.requests[0].summary_source == "auto_generated"


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
    summarizer = RecordingSummarizer(SemanticSummaryResult(summary="## Goal\n- auto QA summary"), [])

    result = auto_compact_agent_session(
        state.session_dir,
        "qa_agent",
        AutoCompactionPolicy(new_message_threshold=2),
        summarizer=summarizer,
    )

    summary = json.loads((state.session_dir / "agent_sessions" / "qa_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    compactions = _jsonl(state.session_dir / "agent_sessions" / "qa_agent" / "compactions.jsonl")
    assert replayed.status == "succeeded"
    assert result.status == "succeeded"
    assert result.compact_status == "auto_compacted"
    assert summary["compact_mode"] == "auto"
    assert summary["compact_id"] == "auto-qa_agent-3"
    assert summary["raw_message_count"] == 3
    assert summary["summary_source"] == "llm_semantic"
    assert compactions[-1]["compact_id"] == "auto-qa_agent-3"


def test_auto_compaction_blocks_when_manual_summary_was_not_replayed(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "ml_agent", "user", "seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="ml_agent", compact_id="manual-mdn-001", summary="manual mdn summary"),
    )
    append_agent_message(state.session_dir, "ml_agent", "assistant", "candidate")

    result = auto_compact_agent_session(
        state.session_dir,
        "ml_agent",
        AutoCompactionPolicy(new_message_threshold=1),
    )

    errors = _jsonl(state.session_dir / "agent_sessions" / "ml_agent" / "compact_errors.jsonl")
    summary = json.loads((state.session_dir / "agent_sessions" / "ml_agent" / "compact_summary.json").read_text(encoding="utf-8"))
    assert result.status == "blocked"
    assert result.blocker == "manual_replay_required"
    assert errors[-1]["blocker"] == "manual_replay_required"
    assert summary["compact_id"] == "manual-mdn-001"


def test_agent_message_append_preserves_append_only_log_until_boundary(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "research_agent", "user", "seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="research_agent", compact_id="manual-rg-001", summary="source seed"),
    )
    replay_agent_compaction(state.session_dir, "research_agent")

    for index in range(AutoCompactionPolicy().new_message_threshold):
        append_agent_message(state.session_dir, "research_agent", "user", f"source update {index}")

    summary_path = state.session_dir / "agent_sessions" / "research_agent" / "compact_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["compact_mode"] == "manual"
    assert summary["compact_id"] == "manual-rg-001"
    assert summary["raw_message_count"] == 1


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
