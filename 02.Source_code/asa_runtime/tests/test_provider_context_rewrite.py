from __future__ import annotations

import pytest
from pathlib import Path

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession, ModelToolChoiceBlocked
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.agents_sdk_runtime.provider_transport import ProviderApiProtocol, provider_transport_request
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap


def test_provider_payload_matrix_rewrites_compacted_messages(tmp_path: Path) -> None:
    session = _compacted_session(tmp_path)

    responses = provider_transport_request(_with_protocol(session, "openai_responses"), ())
    chat = provider_transport_request(_with_protocol(session, "openai_chat_completions"), ())
    anthropic = provider_transport_request(_with_protocol(session, "anthropic_messages"), ())
    gemini = provider_transport_request(_with_protocol(session, "gemini_generate_content"), ())

    assert _dump(responses.payload["input"]).count("COMPACT_SUMMARY_ONCE") == 0
    assert responses.payload["instructions"].count("COMPACT_SUMMARY_ONCE") == 1
    assert "OLD_RAW_SHOULD_NOT_LEAK" not in _dump(responses.payload)
    assert "TAIL_SHOULD_REMAIN" in _dump(responses.payload["input"])
    assert "LATEST_USER_TURN" in _dump(responses.payload["input"])
    assert responses.payload["metadata"]["first_kept_message_sequence"] == "20"
    assert responses.payload["metadata"]["provider_messages_rewritten"] == "true"
    assert "OLD_RAW_SHOULD_NOT_LEAK" not in _dump(chat.payload)
    assert "TAIL_SHOULD_REMAIN" in _dump(chat.payload["messages"])
    assert chat.payload["messages"][0]["content"].count("COMPACT_SUMMARY_ONCE") == 1
    assert "OLD_RAW_SHOULD_NOT_LEAK" not in _dump(anthropic.payload)
    assert anthropic.payload["system"].count("COMPACT_SUMMARY_ONCE") == 1
    assert "TAIL_SHOULD_REMAIN" in _dump(anthropic.payload["messages"])
    assert "OLD_RAW_SHOULD_NOT_LEAK" not in _dump(gemini.payload)
    assert _dump(gemini.payload["systemInstruction"]).count("COMPACT_SUMMARY_ONCE") == 1
    assert "TAIL_SHOULD_REMAIN" in _dump(gemini.payload["contents"])


def test_invalid_compaction_blocks_before_provider_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _compacted_session(tmp_path, blocker="compact_summary_poisoned")
    request_count = {"value": 0}

    def post_with_retry(
        _model: ProviderToolChoiceModel,
        _url: str,
        _payload: JsonMap,
        _token: str | None,
        _protocol: ProviderApiProtocol,
    ) -> JsonMap:
        request_count["value"] += 1
        return {"output": []}

    monkeypatch.setattr(ProviderToolChoiceModel, "_post_with_retry", post_with_retry)
    model = ProviderToolChoiceModel(api_key="test-token")

    with pytest.raises(ModelToolChoiceBlocked, match="compact_summary_poisoned"):
        model.choose_tools(session, ())

    assert request_count["value"] == 0


def _compacted_session(tmp_path: Path, *, blocker: str = "") -> AsaAgentSession:
    return AsaAgentSession(
        run_id="provider-context-rewrite",
        session_id="provider-context-rewrite-session",
        agent_id="orchestrator",
        user_goal="LATEST_USER_TURN",
        endpoint=_endpoint("openai_responses"),
        output_dir=tmp_path,
        registry=default_tool_registry(),
        compact_summary="COMPACT_SUMMARY_ONCE",
        raw_message_count=4,
        provider_context_blocker=blocker,
        compaction_metadata={
            "compact_id": "compact-test",
            "compact_mode": "manual",
            "summary_source": "manual_supplied",
            "first_kept_message_sequence": 20,
            "summary_cutoff_message_sequence": 19,
            "raw_message_count": 4,
            "provider_visible_message_count": 2,
            "rewrite_active": True,
        },
        messages=[
            {"role": "user", "content": "OLD_RAW_SHOULD_NOT_LEAK", "sequence": 18},
            {"role": "assistant", "content": "TAIL_SHOULD_REMAIN", "sequence": 20},
        ],
    )


def _with_protocol(session: AsaAgentSession, protocol: str) -> AsaAgentSession:
    return AsaAgentSession(
        run_id=session.run_id,
        session_id=session.session_id,
        agent_id=session.agent_id,
        user_goal=session.user_goal,
        endpoint=_endpoint(protocol),
        output_dir=session.output_dir,
        registry=session.registry,
        compact_summary=session.compact_summary,
        raw_message_count=session.raw_message_count,
        provider_context_blocker=session.provider_context_blocker,
        compaction_metadata=session.compaction_metadata,
        messages=list(session.messages),
    )


def _endpoint(protocol: str) -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://api.openai.com/v1",
            "auth_mode": "api_key",
            "api_key_env": "MODEL_TOKEN",
            "api_protocol": protocol,
        }
    )


def _dump(value: JsonMap | list[JsonMap]) -> str:
    return repr(value)

