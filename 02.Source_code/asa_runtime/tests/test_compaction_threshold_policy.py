from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    auto_compact_agent_session,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_boundary import provider_boundary_compaction_blocker
from sim_agent.agent_runtime.compaction_provider_summarizer import ProviderSemanticSummarizer
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.cli.tui_state import initial_state
from sim_agent.runtime_config import RUNTIME_CONFIG_ENV


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_valid_threshold_percent_from_config_is_used_for_auto_compaction_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "md_agent", "user", "seed")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="md_agent", compact_id="manual-md-001", summary=""),
        summarizer=RecordingSummarizer(SemanticSummaryResult(summary="compact summary"), []),
        policy=AutoCompactionPolicy(context_window_tokens=100, keep_recent_tokens=16),
    )
    replay_agent_compaction(state.session_dir, "md_agent")
    append_agent_message(state.session_dir, "md_agent", "user", "x" * 180)
    config_path = tmp_path / "runtime-config.json"
    config_path.write_text(
        json.dumps(
            {
                "compaction": {
                    "threshold_percent": 55,
                    "context_window_tokens": 300,
                    "keep_recent_tokens": 16,
                },
            },
        ),
        encoding="utf-8",
    )
    gateway_posts = {"count": 0}

    def summarize(self: ProviderSemanticSummarizer, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        gateway_posts["count"] += 1
        return SemanticSummaryResult(summary=f"auto summary {request.compact_id}")

    monkeypatch.setenv(RUNTIME_CONFIG_ENV, str(config_path))
    monkeypatch.setattr(ProviderSemanticSummarizer, "summarize", summarize)
    handle = load_agent_registry(state.session_dir).handles["md_agent"]

    blocker = provider_boundary_compaction_blocker(state.session_dir, handle)

    summary = json.loads(
        (state.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json").read_text(
            encoding="utf-8",
        ),
    )
    assert blocker == ""
    assert gateway_posts["count"] == 1
    assert summary["compact_mode"] == "auto"
    assert summary["compact_id"] == "auto-md_agent-2"


def test_invalid_threshold_percent_blocks_before_prompt_manifest_and_gateway_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for threshold_percent in (0, 100, -1):
        state = initial_state(tmp_path / f"direct-{threshold_percent}")
        append_agent_message(state.session_dir, "qa_agent", "user", "x" * 500)
        summarizer = RecordingSummarizer(SemanticSummaryResult(summary="must not be requested"), [])

        result = auto_compact_agent_session(
            state.session_dir,
            "qa_agent",
            AutoCompactionPolicy(
                context_window_tokens=100,
                threshold_percent=threshold_percent,
                keep_recent_tokens=16,
            ),
            summarizer=summarizer,
        )

        summary_path = state.session_dir / "agent_sessions" / "qa_agent" / "compact_summary.json"
        assert result.status == "blocked"
        assert result.blocker == "invalid_compaction_threshold_percent"
        assert summarizer.requests == []
        assert not summary_path.exists()

    state = initial_state(tmp_path / "nonnumeric-config")
    append_agent_message(state.session_dir, "md_agent", "user", "x" * 500)
    config_path = tmp_path / "runtime-config.json"
    config_path.write_text(json.dumps({"compaction": {"threshold_percent": "55"}}), encoding="utf-8")
    gateway_posts = {"count": 0}

    def summarize(self: ProviderSemanticSummarizer, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        gateway_posts["count"] += 1
        return SemanticSummaryResult(summary=f"unexpected {request.compact_id}")

    monkeypatch.setenv(RUNTIME_CONFIG_ENV, str(config_path))
    monkeypatch.setattr(ProviderSemanticSummarizer, "summarize", summarize)
    handle = load_agent_registry(state.session_dir).handles["md_agent"]

    blocker = provider_boundary_compaction_blocker(state.session_dir, handle)

    assert blocker == "invalid_compaction_threshold_percent"
    assert gateway_posts["count"] == 0
    assert not list(state.session_dir.rglob("prompt_assembly_manifest.json"))
    assert not (state.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json").exists()


def test_unknown_context_window_blocks_with_context_window_unknown_reason(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_agent_message(state.session_dir, "ml_agent", "user", "x" * 500)
    summarizer = RecordingSummarizer(SemanticSummaryResult(summary="must not be requested"), [])

    result = auto_compact_agent_session(
        state.session_dir,
        "ml_agent",
        AutoCompactionPolicy(
            provider="unregistered-provider",
            model="unregistered-model",
            threshold_percent=55,
            keep_recent_tokens=16,
        ),
        summarizer=summarizer,
    )

    assert result.status == "blocked"
    assert result.blocker == "context_window_unknown"
    assert summarizer.requests == []
