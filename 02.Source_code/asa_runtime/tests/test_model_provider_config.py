from __future__ import annotations

import pytest

from sim_agent.llm_endpoints.config import ModelProviderConfig
from sim_agent.llm_endpoints.config import ModelPolicyError
from sim_agent.llm_endpoints.model_catalog import find_model_catalog_entry


def test_model_provider_config_accepts_normalized_provider_contract_fields() -> None:
    config = ModelProviderConfig.from_mapping(
        {
            "provider_id": "openai",
            "model": "gpt-5.5",
            "api_protocol": "responses",
            "auth_mode": "api_key",
            "reasoning_effort": "max",
            "thinking_mode": "enabled",
            "base_url": "https://api.openai.com/v1/",
            "streaming": True,
            "tool_choice_support": True,
            "provider_session_support": True,
            "credential_source": "api_key_env",
        }
    )

    assert config.provider == "openai"
    assert config.provider_id == "openai"
    assert config.api_protocol == "responses"
    assert config.reasoning_effort == "max"
    assert config.thinking_mode == "enabled"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.streaming is True
    assert config.tool_choice_support is True
    assert config.provider_session_support is True
    assert config.credential_source == "api_key_env"


def test_model_provider_config_sdk_spec_includes_normalized_provider_contract_fields() -> None:
    config = ModelProviderConfig.from_mapping(
        {
            "provider": "oauth_gateway",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "base_url": "http://127.0.0.1:8787/v1",
            "auth_mode": "gateway",
            "credential_source": "gateway_token",
        }
    )

    spec = config.to_agents_sdk_model_spec()

    assert spec.provider == "oauth_gateway"
    assert spec.provider_id == "oauth_gateway"
    assert spec.api_protocol == "openai_compatible"
    assert spec.reasoning_effort == "xhigh"
    assert spec.credential_source == "gateway_token"


def test_model_provider_config_accepts_canonical_transport_protocol_metadata() -> None:
    config = ModelProviderConfig.from_mapping(
        {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "api_protocol": "openai_codex_responses",
            "reasoning_effort": "high",
            "base_url": "https://chatgpt.com/backend-api",
            "auth_mode": "oauth",
        }
    )

    assert config.api_protocol == "openai_codex_responses"
    assert config.to_agents_sdk_model_spec().api_protocol == "openai_codex_responses"


def test_model_provider_config_rejects_malformed_api_protocol() -> None:
    with pytest.raises(ModelPolicyError, match="invalid_api_protocol=not_real"):
        ModelProviderConfig.from_mapping(
            {
                "provider": "openai",
                "model": "gpt-5.5",
                "api_protocol": "not_real",
                "reasoning_effort": "high",
                "base_url": "https://api.openai.com/v1",
                "auth_mode": "api_key",
            }
        )


def test_model_catalog_entries_expose_normalized_provider_contract_fields() -> None:
    entry = find_model_catalog_entry("openai/gpt-5.5")

    assert entry is not None
    assert entry.provider == "openai"
    assert entry.provider_id == "openai"
    assert entry.api_protocol == "responses"
    assert entry.reasoning_effort == "xhigh"
    assert entry.thinking_mode == "enabled"
    assert entry.streaming is True
    assert entry.tool_choice_support is True
    assert entry.provider_session_support is True
    assert entry.credential_source == "api_key_env"
