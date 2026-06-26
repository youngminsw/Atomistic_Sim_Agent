from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from sim_agent.agent_runtime import AutoCompactionPolicy, append_agent_message, auto_compact_agent_session
from sim_agent.agent_runtime.compaction_policy import COMPACT_SCHEMA_VERSION
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.agent_runtime.compaction_store import atomic_write_json
from sim_agent.cli.tui_state import initial_state


SUMMARY_SENTINEL = "BOUNDARY_COMPACT_SUMMARY"
TAIL_SENTINEL = "BOUNDARY_RECENT_TAIL_VISIBLE_ONLY_WHEN_VALID"


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_compaction_state_write_is_atomic_and_idempotent_under_repeated_boundary_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_version": COMPACT_SCHEMA_VERSION, "summary": SUMMARY_SENTINEL}
    summary_path = tmp_path / "compact_summary.json"
    original_replace = Path.replace
    interrupted = {"done": False}

    def fail_first_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".compact_summary.json.") and not interrupted["done"]:
            interrupted["done"] = True
            raise OSError("simulated interrupted atomic rename")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)
    with pytest.raises(OSError, match="simulated interrupted atomic rename"):
        atomic_write_json(summary_path, payload)

    assert not summary_path.exists()
    assert list(tmp_path.glob(".compact_summary.json.*.tmp")) == []

    atomic_write_json(summary_path, payload)
    atomic_write_json(summary_path, payload)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".compact_summary.json.*.tmp")) == []

    state = initial_state(tmp_path / "session")
    for index in range(40):
        append_agent_message(
            state.session_dir,
            "md_agent",
            "assistant" if index % 2 else "user",
            TAIL_SENTINEL if index == 39 else f"boundary idempotency filler {index}",
        )
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary=f"## Goal\n- {SUMMARY_SENTINEL}", short_summary="Boundary summary"),
        [],
    )
    policy = AutoCompactionPolicy(
        context_window_tokens=100,
        threshold_tokens=1,
        keep_recent_tokens=64,
    )

    first = auto_compact_agent_session(state.session_dir, "md_agent", policy, summarizer=summarizer)
    ledger_path = state.session_dir / "agent_sessions" / "md_agent" / "compactions.jsonl"
    ledger_after_first = ledger_path.read_text(encoding="utf-8").splitlines()
    second = auto_compact_agent_session(state.session_dir, "md_agent", policy, summarizer=summarizer)

    assert first.status == "succeeded"
    assert first.compact_status == "auto_compacted"
    assert second.status == "skipped"
    assert second.compact_status == "already_compacted"
    assert ledger_path.read_text(encoding="utf-8").splitlines() == ledger_after_first
    assert len(summarizer.requests) == 1
