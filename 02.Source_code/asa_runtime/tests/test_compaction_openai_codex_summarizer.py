from __future__ import annotations

from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap
import sim_agent.agent_runtime.compaction_provider_summarizer as provider_summarizer


def test_openai_codex_semantic_summarizer_uses_streaming_responses_payload(
    monkeypatch,
) -> None:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://chatgpt.com/backend-api",
            "auth_mode": "oauth",
            "api_key_env": "OPENAI_CODEX_API_KEY",
            "api_protocol": "openai_responses",
        }
    )
    captured_payloads: list[JsonMap] = []

    def gateway_post_json(
        _url: str,
        payload: JsonMap,
        _token: str | None,
        _timeout_s: float,
        _headers: JsonMap,
    ) -> tuple[int, JsonMap]:
        captured_payloads.append(payload)
        return 200, {"output_text": f"codex summary {len(captured_payloads)}"}

    monkeypatch.setattr(provider_summarizer, "gateway_post_json", gateway_post_json)

    result = provider_summarizer.ProviderSemanticSummarizer(endpoint, api_key="test-token").summarize(
        SemanticSummaryRequest(
            agent_id="md_agent",
            compact_id="codex-live-payload",
            compact_mode="auto",
            summary_source="provider_generated",
            system_prompt="system",
            prompt="summarize old context",
            messages_to_summarize=(),
            turn_prefix_messages=(),
            retained_messages=(),
            previous_summary="",
        )
    )

    assert result.summary == "codex summary 1"
    assert result.short_summary == "codex summary 2"
    assert len(captured_payloads) == 2
    assert all(payload["store"] is False for payload in captured_payloads)
    assert all(payload["stream"] is True for payload in captured_payloads)
