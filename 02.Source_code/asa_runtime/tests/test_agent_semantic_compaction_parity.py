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
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_semantic import (
    SemanticSummaryRequest,
    SemanticSummaryResult,
)
from sim_agent.agent_runtime.provider_context_projection import provider_visible_agent_context
from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_transport import provider_transport_request
from sim_agent.cli.tui_state import initial_state
from sim_agent.llm_endpoints import ModelProviderConfig


OLD_RAW_SENTINEL = "SEMANTIC_OLD_RAW_MUST_STAY_ON_DISK_ONLY"
TAIL_SENTINEL = "SEMANTIC_RECENT_TAIL_STAYS_VISIBLE"
CURRENT_SENTINEL = "SEMANTIC_CURRENT_TURN_STAYS_VISIBLE"
SUMMARY_SENTINEL = "SEMANTIC_SUMMARY_REPLACES_OLD_CONTEXT"


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_manual_compaction_uses_gajae_semantic_summary_prompt_and_preserves_raw_log(
    tmp_path: Path,
) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "md_agent", "user", OLD_RAW_SENTINEL)
    append_agent_message(
        state.session_dir,
        "md_agent",
        "assistant",
        'read(path="02.Source_code/input.md") write(path="02.Source_code/output.py")',
    )
    for index in range(28):
        role = "assistant" if index % 2 else "user"
        content = TAIL_SENTINEL if index == 27 else f"semantic tail filler {index}"
        append_agent_message(state.session_dir, "md_agent", role, content)
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(
            summary=f"## Goal\n- {SUMMARY_SENTINEL}",
            short_summary="I compacted old MD context into a semantic checkpoint.",
            preserve_data={
                "openaiRemoteCompaction": {
                    "provider": "openai",
                    "replacementHistory": [{"type": "message", "role": "user"}],
                    "compactionItem": {"type": "compaction_summary", "summary": SUMMARY_SENTINEL},
                }
            },
        ),
        [],
    )

    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id="md_agent",
            compact_id="semantic-md-001",
            summary="",
            summary_source="manual_generated",
        ),
        summarizer=summarizer,
    )
    replayed = replay_agent_compaction(state.session_dir, "md_agent")

    agent_dir = state.session_dir / "agent_sessions" / "md_agent"
    payload = json.loads((agent_dir / "compact_summary.json").read_text(encoding="utf-8"))
    raw_messages = (agent_dir / "messages.jsonl").read_text(encoding="utf-8")
    assert compacted.status == "succeeded"
    assert replayed.status == "succeeded"
    assert payload["schema_version"] == "asa_agent_compact_summary_v4"
    assert payload["summary_source"] == "llm_semantic"
    assert SUMMARY_SENTINEL in payload["summary"]
    assert "<read-files>" in payload["summary"]
    assert "<modified-files>" in payload["summary"]
    assert payload["short_summary"] == "I compacted old MD context into a semantic checkpoint."
    assert payload["semantic_prompt_contract"]["kind"] == "gajae_compaction"
    assert payload["semantic_prompt_contract"]["system_prompt_sha256"]
    assert payload["semantic_prompt_contract"]["summary_prompt_sha256"]
    assert payload["semantic_details"]["readFiles"] == ["02.Source_code/input.md"]
    assert payload["semantic_details"]["modifiedFiles"] == ["02.Source_code/output.py"]
    assert "openaiRemoteCompaction" in payload["preserve_data"]
    assert OLD_RAW_SENTINEL in raw_messages
    assert len(summarizer.requests) == 1
    request = summarizer.requests[0]
    assert "Summarize conversations between users and AI coding assistants" in request.system_prompt
    assert "<conversation>" in request.prompt
    assert "structured context checkpoint handoff summary" in request.prompt
    assert OLD_RAW_SENTINEL in request.prompt


def test_provider_payload_matrix_excludes_old_raw_after_semantic_projection(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "qa_agent", "user", OLD_RAW_SENTINEL)
    for index in range(29):
        role = "assistant" if index % 2 else "user"
        append_agent_message(state.session_dir, "qa_agent", role, TAIL_SENTINEL if index == 28 else f"qa tail {index}")
    summarizer = RecordingSummarizer(
        SemanticSummaryResult(summary=f"## Goal\n- {SUMMARY_SENTINEL}", short_summary="I summarized QA context."),
        [],
    )
    compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id="qa_agent",
            compact_id="semantic-qa-001",
            summary="",
            summary_source="manual_generated",
        ),
        summarizer=summarizer,
    )
    replay_agent_compaction(state.session_dir, "qa_agent")
    handle = load_agent_registry(state.session_dir).handles["qa_agent"]
    context = provider_visible_agent_context(handle)

    for provider, base_url, protocol in (
        ("openai", "https://api.openai.com/v1", "openai_responses"),
        ("openai", "https://api.openai.com/v1", "openai_chat_completions"),
        ("anthropic", "https://api.anthropic.com/v1", "anthropic_messages"),
        ("google-gemini-cli", "https://generativelanguage.googleapis.com", "gemini_generate_content"),
    ):
        request = provider_transport_request(
            _provider_session(tmp_path, provider, base_url, protocol, context),
            (),
        )
        body = json.dumps(request.payload, sort_keys=True)
        assert OLD_RAW_SENTINEL not in body
        assert SUMMARY_SENTINEL in body
        assert TAIL_SENTINEL in body
        assert CURRENT_SENTINEL in body
        assert "Another language model started to solve this problem" in body
        assert "previous_response_id" not in body


def test_append_only_log_does_not_auto_compact_until_provider_boundary(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    summary_path = state.session_dir / "agent_sessions" / "research_agent" / "compact_summary.json"

    for index in range(AutoCompactionPolicy().new_message_threshold):
        append_agent_message(state.session_dir, "research_agent", "user", f"research update {index}")

    assert not summary_path.exists()
    result = auto_compact_agent_session(
        state.session_dir,
        "research_agent",
        AutoCompactionPolicy(new_message_threshold=1),
        summarizer=RecordingSummarizer(
            SemanticSummaryResult(summary="## Goal\n- auto semantic checkpoint", short_summary="I compacted research context."),
            [],
        ),
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result.status == "succeeded"
    assert payload["compact_mode"] == "auto"
    assert payload["summary_source"] == "llm_semantic"


def _provider_session(
    tmp_path: Path,
    provider: str,
    base_url: str,
    protocol: str,
    context,
) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": base_url,
            "auth_mode": "gateway",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
            "api_protocol": protocol,
        }
    )
    return AsaAgentSession(
        run_id="semantic-provider-matrix",
        session_id="semantic-provider-session",
        agent_id="qa_agent",
        user_goal=CURRENT_SENTINEL,
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
        compact_summary=context.compact_summary,
        raw_message_count=context.raw_message_count,
        compaction_metadata=context.compaction.to_json() if context.compaction is not None else {},
        messages=list(context.messages),
    )
