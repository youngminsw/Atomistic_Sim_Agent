from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.agent_runtime.provider_context_projection import provider_visible_agent_context
from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.agents_sdk_runtime.provider_transport import ProviderApiProtocol
from sim_agent.cli.tui_state import initial_state
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap
import sim_agent.agent_runtime.compaction_provider_summarizer as provider_summarizer


AUTH_HEADER_SECRET = "Authorization: Bearer sk-live-auth-secret-1234567890"
BEARER_SECRET = "Bearer sk-direct-bearer-secret-0987654321"
ENV_SECRET = "OPENAI_API_KEY=sk-env-secret-abcdef123456"
JSON_SECRET = '"api_key": "sk-json-secret-fedcba654321"'
TOKEN_SECRET = "MODEL_TOKEN=mt-secret-aaaabbbbcccc"
RAW_SECRET_FRAGMENTS = (
    "sk-live-auth-secret-1234567890",
    "sk-direct-bearer-secret-0987654321",
    "sk-env-secret-abcdef123456",
    "sk-json-secret-fedcba654321",
    "mt-secret-aaaabbbbcccc",
)
VISIBLE_TAIL = "SECRET_REDACTION_TAIL_REMAINS_VISIBLE"
CURRENT_TURN = "SECRET_REDACTION_CURRENT_TURN"
SAFE_SUMMARY = "Secret-bearing setup was compacted with credentials redacted."
REDACTION_MARKER = "[REDACTED_SECRET]"


@dataclass(slots=True)
class EchoingSecretSummarizer:
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return SemanticSummaryResult(
            summary=(
                f"## Goal\n- {SAFE_SUMMARY}\n"
                f"- malicious echo: {AUTH_HEADER_SECRET}\n"
                f"- env echo: {ENV_SECRET}\n"
                f"- json echo: {JSON_SECRET}"
            ),
            short_summary=f"short echo {BEARER_SECRET} {TOKEN_SECRET}",
            preserve_data={
                "openaiRemoteCompaction": {
                    "secretEcho": TOKEN_SECRET,
                    "authorization": AUTH_HEADER_SECRET,
                }
            },
        )


def test_semantic_compaction_redacts_secret_like_content_before_persisting_or_replaying(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = initial_state(tmp_path / "session")
    summarizer = EchoingSecretSummarizer([])
    _append_secret_history(state.session_dir)

    compacted = compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="md_agent", compact_id="secret-redaction-001"),
        summarizer=summarizer,
        policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
    )
    replayed = replay_agent_compaction(state.session_dir, "md_agent")

    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    summary_payload = json.loads((handle.session_dir / "compact_summary.json").read_text(encoding="utf-8"))
    raw_messages = handle.messages_path.read_text(encoding="utf-8")
    context = provider_visible_agent_context(handle)
    transport_payloads: list[JsonMap] = []

    def post_with_retry(
        _model: ProviderToolChoiceModel,
        _url: str,
        payload: JsonMap,
        _token: str | None,
        _protocol: ProviderApiProtocol,
    ) -> JsonMap:
        transport_payloads.append(payload)
        return {"output": []}

    monkeypatch.setattr(ProviderToolChoiceModel, "_post_with_retry", post_with_retry)
    ProviderToolChoiceModel(api_key="test-token").choose_tools(_provider_session(tmp_path, context), ())
    manifest = json.loads((tmp_path / "prompt_assembly_manifest.json").read_text(encoding="utf-8"))

    assert compacted.status == "succeeded"
    assert replayed.status == "succeeded"
    assert summarizer.requests
    assert SAFE_SUMMARY in summary_payload["summary"]
    assert REDACTION_MARKER in json.dumps(summary_payload, sort_keys=True)
    assert all(secret in raw_messages for secret in RAW_SECRET_FRAGMENTS)

    replay_surfaces = (
        summarizer.requests[0].prompt,
        json.dumps(summarizer.requests[0].messages_to_summarize, sort_keys=True),
        json.dumps(summary_payload, sort_keys=True),
        context.compact_summary,
        json.dumps(context.compaction.to_json() if context.compaction else {}, sort_keys=True),
        json.dumps(transport_payloads, sort_keys=True),
        json.dumps(manifest, sort_keys=True),
    )
    for surface in replay_surfaces:
        for secret in RAW_SECRET_FRAGMENTS:
            assert secret not in surface


def test_secret_like_compaction_input_blocks_or_redacts_before_provider_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = initial_state(tmp_path / "session")
    summarizer = EchoingSecretSummarizer([])
    _append_secret_history(state.session_dir)
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="md_agent", compact_id="secret-redaction-002"),
        summarizer=summarizer,
        policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
    )
    replay_agent_compaction(state.session_dir, "md_agent")
    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    context = provider_visible_agent_context(handle)
    post_called = {"value": False}

    def post_with_retry(
        _model: ProviderToolChoiceModel,
        _url: str,
        payload: JsonMap,
        _token: str | None,
        _protocol: ProviderApiProtocol,
    ) -> JsonMap:
        post_called["value"] = True
        payload_dump = json.dumps(payload, sort_keys=True)
        for secret in RAW_SECRET_FRAGMENTS:
            assert secret not in payload_dump
        return {"output": []}

    monkeypatch.setattr(ProviderToolChoiceModel, "_post_with_retry", post_with_retry)
    ProviderToolChoiceModel(api_key="test-token").choose_tools(_provider_session(tmp_path, context), ())

    assert post_called["value"] is True


def test_provider_short_summary_request_redacts_secret_echo_before_second_gateway_post(
    monkeypatch,
) -> None:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://api.openai.com/v1",
            "auth_mode": "api_key",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
            "api_protocol": "openai_chat_completions",
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
        if len(captured_payloads) == 1:
            return 200, {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Provider summary echoed a credential "
                                "Authorization: Bearer sk-leaked-second-call-12345"
                            )
                        }
                    }
                ]
            }
        return 200, {"choices": [{"message": {"content": "redacted short summary"}}]}

    monkeypatch.setenv("MODEL_GATEWAY_TOKEN", "test-token")
    monkeypatch.setattr(provider_summarizer, "gateway_post_json", gateway_post_json)

    result = provider_summarizer.ProviderSemanticSummarizer(endpoint).summarize(
        SemanticSummaryRequest(
            agent_id="md_agent",
            compact_id="secret-second-call",
            compact_mode="manual",
            summary_source="manual_generated",
            system_prompt="system",
            prompt="summarize old context",
            messages_to_summarize=(),
            turn_prefix_messages=(),
            retained_messages=(),
            previous_summary="",
        )
    )

    assert result.short_summary == "redacted short summary"
    assert len(captured_payloads) == 2
    second_payload = json.dumps(captured_payloads[1], sort_keys=True)
    assert "sk-leaked-second-call-12345" not in second_payload
    assert REDACTION_MARKER in second_payload


def _append_secret_history(session_dir: Path) -> None:
    append_agent_message(session_dir, "md_agent", "user", f"configure auth with {AUTH_HEADER_SECRET}")
    append_agent_message(session_dir, "md_agent", "assistant", f"stored temporary token {TOKEN_SECRET}")
    append_agent_message(session_dir, "md_agent", "user", f"debug payload {JSON_SECRET}")
    append_agent_message(session_dir, "md_agent", "assistant", f"curl -H '{BEARER_SECRET}' https://example.test")
    append_agent_message(session_dir, "md_agent", "user", f"env file contains {ENV_SECRET}")
    for index in range(34):
        role = "assistant" if index % 2 else "user"
        content = VISIBLE_TAIL if index == 33 else f"tail filler {index}"
        append_agent_message(session_dir, "md_agent", role, content)


def _provider_session(tmp_path: Path, context) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://api.openai.com/v1",
            "auth_mode": "gateway",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
            "api_protocol": "openai_responses",
        }
    )
    return AsaAgentSession(
        run_id="secret-redaction-provider",
        session_id="secret-redaction-session",
        agent_id="md_agent",
        user_goal=CURRENT_TURN,
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
        compact_summary=context.compact_summary,
        raw_message_count=context.raw_message_count,
        compaction_metadata=context.compaction.to_json() if context.compaction is not None else {},
        messages=list(context.messages),
    )
