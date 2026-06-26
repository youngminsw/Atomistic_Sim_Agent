from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    auto_compact_agent_session,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.agent_runtime.provider_context_projection import provider_visible_agent_context
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.agent_loop import ModelToolChoiceBlocked
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.cli.tui_compaction import handle_compact
from sim_agent.cli.tui_state import initial_state
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap


OLD_RAW_SENTINEL = "ANTI_SLOP_OLD_RAW_MUST_STAY_ON_DISK_ONLY"
TAIL_SENTINEL = "ANTI_SLOP_RECENT_TAIL_STAYS_VISIBLE"
CURRENT_SENTINEL = "ANTI_SLOP_CURRENT_TURN_STAYS_VISIBLE"
SUMMARY_SENTINEL = "ANTI_SLOP_SUMMARY_REPLACES_OLD_CONTEXT"


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_slash_compact_instructions_are_focus_not_manual_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "md_agent", "user", "old context to summarize semantically")
    instruction_text = "prioritize MD setup risks; do not use this text as the summary"
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary=f"## Goal\n- {SUMMARY_SENTINEL}", short_summary="Semantic checkpoint."),
        [],
    )
    policy = AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=64)
    monkeypatch.setattr("sim_agent.cli.tui_compaction._summarizer_for_state", lambda state: summarizer)
    monkeypatch.setattr("sim_agent.cli.tui_compaction._policy_for_state", lambda state: policy)
    stream = StringIO()

    handle_compact(("md_agent", instruction_text), state, stream)

    summary_path = state.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["summary_source"] == "llm_semantic"
    assert payload["summary"] != instruction_text
    assert summarizer.requests[0].additional_focus == instruction_text
    assert "Additional focus:" in summarizer.requests[0].prompt


def test_manual_compaction_instructions_are_added_to_semantic_prompt_not_summary(
    tmp_path: Path,
) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "research_agent", "user", OLD_RAW_SENTINEL)
    instruction_text = "Additional focus: preserve unresolved paper citations."
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary=f"## Goal\n- {SUMMARY_SENTINEL}", short_summary="Semantic checkpoint."),
        [],
    )

    compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id="research_agent",
            compact_id="semantic-rg-instructions",
            summary_source="manual_generated",
            additional_focus=instruction_text,
        ),
        summarizer=summarizer,
    )

    payload = json.loads(
        (state.session_dir / "agent_sessions" / "research_agent" / "compact_summary.json").read_text(
            encoding="utf-8",
        ),
    )
    assert payload["summary"] != instruction_text
    assert "Additional focus:" in summarizer.requests[0].prompt
    assert instruction_text in summarizer.requests[0].prompt


def test_provider_prompt_manifest_excludes_old_raw_after_semantic_projection(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "qa_agent", "user", OLD_RAW_SENTINEL)
    for index in range(29):
        role = "assistant" if index % 2 else "user"
        append_agent_message(state.session_dir, "qa_agent", role, TAIL_SENTINEL if index == 28 else f"qa tail {index}")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id="qa_agent",
            compact_id="semantic-qa-manifest",
            summary="",
            summary_source="manual_generated",
        ),
        summarizer=RecordingSummarizer(
            SemanticSummaryResult(summary=f"## Goal\n- {SUMMARY_SENTINEL}", short_summary="I summarized QA context."),
            [],
        ),
        policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
    )
    replay_agent_compaction(state.session_dir, "qa_agent")
    handle = load_agent_registry(state.session_dir).handles["qa_agent"]
    context = provider_visible_agent_context(handle)
    model = RecordingProviderToolChoiceModel()

    model.complete_turn(
        _provider_session(tmp_path, context.compact_summary, context.compaction.to_json(), list(context.messages)),
        (),
    )

    manifest_text = (tmp_path / "prompt_assembly_manifest.json").read_text(encoding="utf-8")
    assert OLD_RAW_SENTINEL in (handle.session_dir / "messages.jsonl").read_text(encoding="utf-8")
    assert OLD_RAW_SENTINEL not in json.dumps(model.posts, sort_keys=True)
    assert OLD_RAW_SENTINEL not in manifest_text
    assert SUMMARY_SENTINEL in manifest_text


def test_auto_compaction_skips_many_short_messages_below_token_threshold(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary="## Goal\n- should not be used", short_summary="unused"),
        [],
    )
    for index in range(80):
        append_agent_message(state.session_dir, "research_agent", "user", f"ok {index}")

    result = auto_compact_agent_session(
        state.session_dir,
        "research_agent",
        AutoCompactionPolicy(context_window_tokens=10_000, threshold_percent=70, keep_recent_tokens=64),
        summarizer=summarizer,
    )

    assert result.status == "skipped"
    assert result.compact_status == "below_token_threshold"
    assert summarizer.requests == []


def test_auto_compaction_runs_for_few_large_messages_above_token_threshold(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary="## Goal\n- large transcript checkpoint", short_summary="Large checkpoint."),
        [],
    )
    large_message = " ".join(f"token{index}" for index in range(9000))
    append_agent_message(state.session_dir, "research_agent", "user", large_message)
    append_agent_message(state.session_dir, "research_agent", "assistant", large_message)

    result = auto_compact_agent_session(
        state.session_dir,
        "research_agent",
        AutoCompactionPolicy(context_window_tokens=10_000, threshold_percent=70, keep_recent_tokens=200),
        summarizer=summarizer,
    )

    assert result.status == "succeeded"
    assert result.compact_status == "auto_compacted"
    assert len(summarizer.requests) == 1


def test_invalid_compaction_state_blocks_before_provider_post(tmp_path: Path) -> None:
    session = AsaAgentSession(
        run_id="invalid-compact",
        session_id="invalid-compact-session",
        agent_id="qa_agent",
        user_goal=CURRENT_SENTINEL,
        endpoint=_endpoint(),
        output_dir=tmp_path,
        registry=default_tool_registry(),
        compact_summary=f"## Compact summary\n{SUMMARY_SENTINEL}",
        raw_message_count=2,
        compaction_metadata={"first_kept_message_sequence": 2},
        messages=[{"role": "user", "content": OLD_RAW_SENTINEL}],
    )
    model = RecordingProviderToolChoiceModel()

    with pytest.raises(ModelToolChoiceBlocked, match="missing_compaction_message_sequence"):
        model.complete_turn(session, ())

    assert model.posts == []
    assert not (tmp_path / "prompt_assembly_manifest.json").exists()


class RecordingProviderToolChoiceModel(ProviderToolChoiceModel):
    def __init__(self) -> None:
        super().__init__(api_key="test-token", retry_count=0)
        object.__setattr__(self, "posts", [])

    def _post_with_retry(self, url: str, payload: JsonMap, token: str | None, protocol) -> JsonMap:
        self.posts.append({"url": url, "payload": payload, "token": token or "", "protocol": protocol.value})
        return {"output_text": "provider-ok"}


def _provider_session(
    tmp_path: Path,
    compact_summary: str,
    compaction_metadata: JsonMap,
    messages: list[JsonMap],
) -> AsaAgentSession:
    return AsaAgentSession(
        run_id="semantic-provider-manifest",
        session_id="semantic-provider-session",
        agent_id="qa_agent",
        user_goal=CURRENT_SENTINEL,
        endpoint=_endpoint(),
        output_dir=tmp_path,
        registry=default_tool_registry(),
        compact_summary=compact_summary,
        raw_message_count=len(messages),
        compaction_metadata=compaction_metadata,
        messages=messages,
    )


def _endpoint() -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://api.openai.com/v1",
            "auth_mode": "gateway",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
            "api_protocol": "openai_responses",
        },
    )
