from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession, ModelToolChoiceBlocked
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.agents_sdk_runtime.provider_transport import (
    ProviderApiProtocol,
    ProviderTransportPolicyError,
    api_protocol_for_session,
    provider_transport_request,
)
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_provider


def test_unknown_provider_requires_explicit_protocol(tmp_path: Path) -> None:
    session = _session(tmp_path, provider="openai", api_protocol="openai_responses")
    unknown = replace(
        session,
        endpoint=replace(session.endpoint, provider="unknown-provider", api_protocol=""),
    )

    with pytest.raises(ProviderTransportPolicyError, match="unsupported_provider_protocol=unknown-provider"):
        provider_transport_request(unknown, _tools(unknown))


def test_non_builtin_provider_requires_configured_protocol(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="deepseek",
        api_protocol="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    )
    implicit = replace(session, endpoint=replace(session.endpoint, api_protocol=""))

    with pytest.raises(ProviderTransportPolicyError, match="unsupported_provider_protocol=deepseek"):
        api_protocol_for_session(implicit)


def test_missing_explicit_auth_source_blocks_before_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    gateway_calls: list[dict[str, object]] = []
    stored_token_calls: list[str] = []

    def gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        gateway_calls.append(payload)
        return 200, {"output_text": "posted"}

    def access_token_for_provider(provider: str) -> str:
        stored_token_calls.append(provider)
        return "stored-token"

    monkeypatch.setenv("MODEL_TOKEN", "env-token")
    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", gateway_post_json)
    monkeypatch.setattr(provider_tool_choice_model, "access_token_for_provider", access_token_for_provider)
    valid_session = _session(tmp_path)
    session = replace(valid_session, endpoint=replace(valid_session.endpoint, credential_source=""))
    model = ProviderToolChoiceModel(retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="explicit_credential_source_required"):
        model.complete_turn(session, _tools(session))

    assert gateway_calls == []
    assert stored_token_calls == []
    assert not (tmp_path / "prompt_assembly_manifest.json").exists()


def test_gateway_auth_does_not_fall_back_to_stored_oauth_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    gateway_calls: list[dict[str, object]] = []
    stored_token_calls: list[str] = []

    def gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        gateway_calls.append(payload)
        return 200, {"output_text": "posted"}

    def access_token_for_provider(provider: str) -> str:
        stored_token_calls.append(provider)
        return "stored-oauth-token"

    monkeypatch.delenv("MODEL_GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(tmp_path / "credentials.json"))
    login_model_provider(
        {
            "provider": "oauth_gateway",
            "access_token": "stored-oauth-token",
            "refresh_token": "stored-refresh-token",
            "auth_mode": "oauth",
            "expires_in_s": 3600,
        }
    )
    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", gateway_post_json)
    monkeypatch.setattr(provider_tool_choice_model, "access_token_for_provider", access_token_for_provider)
    session = _session(
        tmp_path,
        provider="oauth_gateway",
        auth_mode="gateway",
        credential_source="gateway_token",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )
    model = ProviderToolChoiceModel(retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="missing_gateway_token"):
        model.complete_turn(session, _tools(session))

    assert gateway_calls == []
    assert stored_token_calls == []


def test_oauth_auth_does_not_fall_back_to_env_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    gateway_calls: list[dict[str, object]] = []
    stored_token_calls: list[str] = []

    def gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        gateway_calls.append(payload)
        return 200, {"output_text": "posted"}

    def access_token_for_provider(provider: str) -> None:
        stored_token_calls.append(provider)
        return None

    monkeypatch.setenv("OPENAI_CODEX_TOKEN", "env-oauth-token")
    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", gateway_post_json)
    monkeypatch.setattr(provider_tool_choice_model, "access_token_for_provider", access_token_for_provider)
    session = _session(
        tmp_path,
        provider="openai-codex",
        auth_mode="oauth",
        credential_source="oauth_token",
        api_key_env="OPENAI_CODEX_TOKEN",
        base_url="https://chatgpt.com/backend-api",
    )
    model = ProviderToolChoiceModel(retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="missing_oauth_token"):
        model.complete_turn(session, _tools(session))

    assert gateway_calls == []
    assert stored_token_calls == ["openai-codex"]


def test_explicit_openai_protocol_and_key_source_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    gateway_calls: list[dict[str, object]] = []

    def gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        gateway_calls.append(payload)
        return 200, {"output_text": "No tool needed."}

    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", gateway_post_json)
    session = _session(tmp_path, api_protocol="openai_responses", credential_source="api_key_env")
    model = ProviderToolChoiceModel(api_key="explicit-api-key", retry_count=0)

    result = model.complete_turn(session, _tools(session))

    assert result.final_output == "No tool needed."
    assert len(gateway_calls) == 1
    assert api_protocol_for_session(session) is ProviderApiProtocol.OPENAI_RESPONSES
    manifest = json.loads((tmp_path / "prompt_assembly_manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_protocol"] == "openai_responses"
    assert manifest["credential_source"] == "api_key_env"
    assert manifest["auth_mode"] == "api_key"


def _session(
    tmp_path: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-5.5",
    api_protocol: str = "openai_responses",
    base_url: str = "https://provider.invalid/v1",
    auth_mode: str = "api_key",
    credential_source: str = "api_key_env",
    api_key_env: str = "MODEL_TOKEN",
) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": model,
            "reasoning_effort": "high",
            "api_protocol": api_protocol,
            "base_url": base_url,
            "auth_mode": auth_mode,
            "credential_source": credential_source,
            "api_key_env": api_key_env,
        }
    )
    return AsaAgentSession(
        run_id="provider-hardening-run",
        session_id="provider-hardening-session",
        agent_id="orchestrator",
        user_goal="select a safe tool",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )


def _tools(session: AsaAgentSession) -> tuple[dict[str, object], ...]:
    return tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True)
