from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_runtime import AutoCompactionPolicy, append_agent_message, auto_compact_agent_session
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.compaction_tokens import estimate_messages_tokens
import sim_agent.compaction_tokens as compaction_tokens
from sim_agent.schemas._parse import JsonMap


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_provider_visible_estimate_counts_system_context_tools_and_messages() -> None:
    provider_visible_request = {
        "system_context": "SYSTEM_CONTEXT_VISIBLE_TO_PROVIDER",
        "compact_summary": "COMPACT_SUMMARY_VISIBLE_TO_PROVIDER",
        "role_context": "ROLE_CONTEXT_VISIBLE_TO_PROVIDER",
        "caller_context": "CALLER_CONTEXT_VISIBLE_TO_PROVIDER",
        "workflow_context": "WORKFLOW_CONTEXT_VISIBLE_TO_PROVIDER",
        "project_context": "PROJECT_CONTEXT_VISIBLE_TO_PROVIDER",
        "skill_context": "SKILL_CONTEXT_VISIBLE_TO_PROVIDER",
        "messages": [
            {"role": "user", "content": "RETAINED_MESSAGE_VISIBLE_TO_PROVIDER"},
            {"role": "assistant", "content": "CURRENT_MESSAGE_VISIBLE_TO_PROVIDER"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "visible_tool",
                "description": "TOOL_SCHEMA_VISIBLE_TO_PROVIDER",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
    }

    estimate = compaction_tokens.estimate_provider_visible_tokens(provider_visible_request)

    for surface in (
        "system_context",
        "compact_summary",
        "role_context",
        "caller_context",
        "workflow_context",
        "project_context",
        "skill_context",
        "messages",
        "tools",
    ):
        assert estimate.breakdown[surface] > 0
    assert estimate.total_tokens == sum(estimate.breakdown.values())
    assert estimate.total_tokens > estimate_messages_tokens(provider_visible_request["messages"])


def test_auto_compaction_triggers_when_non_message_context_crosses_threshold(tmp_path: Path) -> None:
    state = _state(tmp_path)
    append_agent_message(state.session_dir, "md_agent", "user", "short visible user turn")
    summarizer = RecordingSummarizer(SemanticSummaryResult(summary="auto summary from provider-visible context"), [])

    result = auto_compact_agent_session(
        state.session_dir,
        "md_agent",
        AutoCompactionPolicy(
            context_window_tokens=2_000,
            threshold_tokens=900,
            keep_recent_tokens=16,
        ),
        summarizer=summarizer,
    )

    assert estimate_messages_tokens(({"role": "user", "content": "short visible user turn"},)) < 900
    assert result.status == "succeeded"
    assert result.compact_status == "auto_compacted"
    assert result.estimated_context_tokens > result.threshold_tokens
    assert summarizer.requests


def test_provider_visible_estimate_supports_openai_chat_anthropic_and_gemini_shapes() -> None:
    messages = [{"role": "user", "content": "MESSAGE_VISIBLE_TO_PROVIDER"}]
    tool = {
        "type": "function",
        "function": {
            "name": "shape_tool",
            "description": "TOOL_VISIBLE_TO_PROVIDER",
            "parameters": {"type": "object"},
        },
    }
    payloads: tuple[JsonMap, ...] = (
        {
            "instructions": "OPENAI_RESPONSES_INSTRUCTIONS_VISIBLE",
            "input": messages,
            "tools": [{"type": "function", "name": "shape_tool", "description": "TOOL_VISIBLE_TO_PROVIDER"}],
        },
        {
            "messages": [{"role": "system", "content": "OPENAI_CHAT_SYSTEM_VISIBLE"}, *messages],
            "tools": [tool],
        },
        {
            "system": "ANTHROPIC_SYSTEM_VISIBLE",
            "messages": messages,
            "tools": [{"name": "shape_tool", "description": "TOOL_VISIBLE_TO_PROVIDER", "input_schema": {"type": "object"}}],
        },
        {
            "systemInstruction": {"parts": [{"text": "GEMINI_SYSTEM_VISIBLE"}]},
            "contents": [{"role": "user", "parts": [{"text": "MESSAGE_VISIBLE_TO_PROVIDER"}]}],
            "tools": [{"functionDeclarations": [{"name": "shape_tool", "description": "TOOL_VISIBLE_TO_PROVIDER"}]}],
        },
    )

    estimates = tuple(compaction_tokens.estimate_provider_visible_tokens(payload) for payload in payloads)

    assert all(estimate.breakdown["messages"] > 0 for estimate in estimates)
    assert all(estimate.breakdown["tools"] > 0 for estimate in estimates)
    assert all(estimate.total_tokens > estimate.breakdown["messages"] for estimate in estimates)


def test_gemini_contents_parts_text_contributes_to_provider_visible_estimate() -> None:
    long_visible_text = "GEMINI_PROVIDER_VISIBLE_TEXT_" * 180
    payload = {
        "systemInstruction": {"parts": [{"text": "GEMINI_SYSTEM_VISIBLE"}]},
        "contents": [{"role": "user", "parts": [{"text": long_visible_text}]}],
    }

    estimate = compaction_tokens.estimate_provider_visible_tokens(payload)

    assert estimate.breakdown["messages"] >= compaction_tokens.estimate_text_tokens(long_visible_text)
    assert estimate.breakdown["messages"] > 1_000


def _state(tmp_path: Path):
    from sim_agent.cli.tui_state import initial_state

    return initial_state(tmp_path)
