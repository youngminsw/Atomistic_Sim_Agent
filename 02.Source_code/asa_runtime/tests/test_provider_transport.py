from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_transport import (
    ProviderApiProtocol,
    ProviderTransportPolicyError,
    api_protocol_for_session,
    provider_transport_request,
    transport_tool_calls,
)
from sim_agent.llm_endpoints import ModelProviderConfig


def test_openai_responses_transport_uses_responses_endpoint(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="openai", base_url="https://api.openai.com/v1")

    request = provider_transport_request(session, _tools(session))

    assert request.protocol is ProviderApiProtocol.OPENAI_RESPONSES
    assert request.url == "https://api.openai.com/v1/responses"
    assert request.payload["model"] == "gpt-5.5"
    assert request.payload["input"] == [{"role": "user", "content": "select a safe tool"}]
    assert request.payload["tool_choice"] == "auto"


def test_openai_codex_transport_uses_chatgpt_codex_endpoint(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="openai-codex", base_url="https://chatgpt.com/backend-api")

    request = provider_transport_request(session, _tools(session))

    assert request.protocol is ProviderApiProtocol.OPENAI_CODEX_RESPONSES
    assert request.url == "https://chatgpt.com/backend-api/codex/responses"
    assert request.payload["model"] == "gpt-5.5"
    assert request.payload["store"] is False
    assert request.payload["stream"] is True
    assert "metadata" not in request.payload


def test_openai_compatible_transport_uses_chat_completions_endpoint(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_protocol="openai_compatible",
    )

    request = provider_transport_request(session, _tools(session))

    assert request.protocol is ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS
    assert request.url == "https://api.deepseek.com/v1/chat/completions"
    assert request.payload["messages"][0]["role"] == "system"
    tool = request.payload["tools"][0]
    assert tool["type"] == "function"
    assert "function" in tool


def test_anthropic_transport_uses_messages_endpoint_and_tool_use_parser(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="anthropic",
        model="claude-sonnet-4.5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
    )

    request = provider_transport_request(session, _tools(session))
    calls = transport_tool_calls(
        request.protocol,
        {"content": [{"type": "tool_use", "name": "artifact_write", "input": {"relative_path": "x", "content": "y"}}]},
    )

    assert request.protocol is ProviderApiProtocol.ANTHROPIC_MESSAGES
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.payload["tool_choice"] == {"type": "auto"}
    assert request.payload["tools"][0]["input_schema"]["type"] == "object"
    assert calls == ({"name": "artifact_write", "arguments": {"relative_path": "x", "content": "y"}},)


def test_gemini_transport_uses_generate_content_endpoint_and_parser(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="google-gemini-cli",
        model="gemini-3-pro-preview",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env="GOOGLE_API_KEY",
    )

    request = provider_transport_request(session, _tools(session))
    calls = transport_tool_calls(
        request.protocol,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "graphdb_dry_run",
                                    "args": {"database_name": "asa"},
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )

    assert request.protocol is ProviderApiProtocol.GEMINI_GENERATE_CONTENT
    assert request.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent"
    assert request.payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert calls == ({"name": "graphdb_dry_run", "arguments": {"database_name": "asa"}},)


def test_ollama_transport_is_openai_compatible_not_responses(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="ollama", model="llama3", base_url="http://localhost:11434/v1", auth_mode="none")

    assert api_protocol_for_session(session) is ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE
    assert provider_transport_request(session, _tools(session)).url == "http://localhost:11434/v1/chat/completions"


def test_provider_transport_rejects_malformed_explicit_protocol(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="openai", base_url="https://api.openai.com/v1")
    malformed = replace(session, endpoint=replace(session.endpoint, api_protocol="definitely_not_a_protocol"))

    with pytest.raises(ProviderTransportPolicyError, match="invalid_api_protocol=definitely_not_a_protocol"):
        provider_transport_request(malformed, _tools(malformed))


def test_provider_transport_exposes_subagent_control_schema(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="openai", base_url="https://api.openai.com/v1")

    request = provider_transport_request(session, _tools(session))

    subagent_control = next(tool for tool in request.payload["tools"] if tool["name"] == "subagent_control")
    assert subagent_control["parameters"]["required"] == ["action"]
    assert "caller_agent" not in subagent_control["parameters"]["properties"]
    assert "pause" in subagent_control["parameters"]["properties"]["action"]["enum"]
    assert "steer" in subagent_control["parameters"]["properties"]["action"]["enum"]


def _session(
    tmp_path: Path,
    *,
    provider: str,
    base_url: str,
    model: str = "gpt-5.5",
    auth_mode: str = "api_key",
    api_key_env: str = "MODEL_TOKEN",
    api_protocol: str | None = None,
) -> AsaAgentSession:
    config = {
        "provider": provider,
        "model": model,
        "reasoning_effort": "high",
        "base_url": base_url,
        "auth_mode": auth_mode,
        "api_key_env": api_key_env,
    }
    if api_protocol is not None:
        config["api_protocol"] = api_protocol
    endpoint = ModelProviderConfig.from_mapping(config)
    return AsaAgentSession(
        run_id="provider-transport-test",
        session_id="provider-transport-session",
        agent_id="orchestrator",
        user_goal="select a safe tool",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )


def _tools(session: AsaAgentSession) -> tuple[dict[str, object], ...]:
    return tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True)
