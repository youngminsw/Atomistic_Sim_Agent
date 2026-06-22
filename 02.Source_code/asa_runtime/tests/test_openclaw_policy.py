from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "config"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_config(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_model_provider_primary_defaults_to_gpt55_high() -> None:
    from sim_agent.llm_endpoints import ModelProviderConfig, ModelUseCase

    config = ModelProviderConfig.from_mapping(_load_config("openclaw_missing_model.json"))
    spec = config.to_agents_sdk_model_spec()

    assert config.use_case == ModelUseCase.PRIMARY_CONTROL
    assert spec.provider == "openclaw"
    assert spec.model == "gpt-5.5"
    assert spec.reasoning_effort == "high"
    assert spec.base_url == "https://openclaw.local/v1"


def test_low_risk_helper_model_can_use_lower_reasoning() -> None:
    from sim_agent.llm_endpoints import ModelProviderConfig, ModelUseCase

    config = ModelProviderConfig.from_mapping(_load_config("openclaw_helper_valid.json"))
    spec = config.to_agents_sdk_model_spec()

    assert config.use_case == ModelUseCase.LOW_RISK_EXTRACTION
    assert spec.provider == "openclaw"
    assert spec.model == "gpt-5.3-codex-spark"
    assert spec.reasoning_effort == "low"
    assert spec.structured_outputs is True


def test_oauth_refresh_placeholder_is_preserved_without_execution() -> None:
    from sim_agent.llm_endpoints import ModelProviderConfig

    config = ModelProviderConfig.from_mapping(_load_config("openclaw_oauth_refresh_valid.json"))
    spec = config.to_agents_sdk_model_spec()

    assert spec.api_key_env == "OPENCLAW_OAUTH_TOKEN"
    assert spec.auth_mode == "oauth"
    assert spec.auth_refresh_command == "openclaw auth token --print"
    assert spec.model == "gpt-5.5"


def test_low_reasoning_is_rejected_for_physics_decision() -> None:
    from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig

    with pytest.raises(ModelPolicyError, match="high_stakes_model_requires_high_reasoning"):
        ModelProviderConfig.from_mapping(_load_config("openclaw_helper_physics_invalid.json"))


def test_direct_openai_endpoint_is_accepted_when_explicitly_configured() -> None:
    from sim_agent.llm_endpoints import ModelProviderConfig

    config = ModelProviderConfig.from_mapping(_load_config("direct_openai_valid.json"))

    assert config.provider == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.auth_mode == "api_key"
    assert config.api_key_env == "OPENAI_API_KEY"


def test_oauth_gateway_endpoint_uses_canonical_gateway_auth_mode() -> None:
    from sim_agent.llm_endpoints import ModelProviderConfig

    config = ModelProviderConfig.from_mapping(_load_config("oauth_gateway_valid.json"))
    spec = config.to_agents_sdk_model_spec()

    assert config.provider == "oauth_gateway"
    assert config.auth_mode == "gateway"
    assert spec.auth_mode == "gateway"
    assert spec.api_key_env == "MODEL_GATEWAY_TOKEN"
    assert spec.auth_refresh_command == "model-gateway auth refresh --print"


def test_gateway_token_is_rejected_as_legacy_auth_mode() -> None:
    from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig

    with pytest.raises(ModelPolicyError, match="invalid_auth_mode=gateway_token"):
        ModelProviderConfig.from_mapping(_load_config("gateway_token_invalid.json"))


def test_model_provider_config_cli_accepts_valid_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "check_model_provider_config.py"),
            "--config",
            str(FIXTURE_ROOT / "openclaw_valid.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_provider_config=true" in result.stdout
    assert "primary_model=gpt-5.5" in result.stdout
    assert "reasoning=high" in result.stdout


def test_model_provider_config_cli_accepts_explicit_direct_openai() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "check_model_provider_config.py"),
            "--config",
            str(FIXTURE_ROOT / "direct_openai_valid.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_provider_config=true" in result.stdout
    assert "provider=openai" in result.stdout
    assert "auth_mode=api_key" in result.stdout
    assert "api_key_env=OPENAI_API_KEY" in result.stdout


def test_model_provider_config_cli_accepts_canonical_gateway_auth() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "check_model_provider_config.py"),
            "--config",
            str(FIXTURE_ROOT / "oauth_gateway_valid.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_provider_config=true" in result.stdout
    assert "provider=oauth_gateway" in result.stdout
    assert "auth_mode=gateway" in result.stdout
    assert "auth_refresh_configured=true" in result.stdout


def test_model_provider_config_cli_rejects_legacy_gateway_token_auth() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "check_model_provider_config.py"),
            "--config",
            str(FIXTURE_ROOT / "gateway_token_invalid.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "invalid_auth_mode=gateway_token" in result.stdout
