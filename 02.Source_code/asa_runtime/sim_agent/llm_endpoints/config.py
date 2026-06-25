from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse, urlunparse

from sim_agent.provider_registry import LEGACY_MODEL_GATEWAY_TOKEN_ENV, default_api_key_env, default_auth_mode, provider_ids
from sim_agent.schemas._parse import JsonMap, as_mapping, str_field
from sim_agent.schemas.errors import ProviderConfigPolicyError

from .policy import is_allowed_openclaw_base_url, normalize_openclaw_base_url


PRIMARY_MODEL = "gpt-5.5"
PRIMARY_REASONING = "high"
SUPPORTED_PROVIDERS = frozenset(provider_ids(include_legacy=True))
DEFAULT_API_KEY_ENV_BY_PROVIDER = {
    provider: default_api_key_env(provider) for provider in SUPPORTED_PROVIDERS
}
DEFAULT_API_KEY_ENV_BY_PROVIDER.update(
    {
        "openclaw": "OPENCLAW_OAUTH_TOKEN",
        "oauth_gateway": LEGACY_MODEL_GATEWAY_TOKEN_ENV,
        "anthropic_gateway": LEGACY_MODEL_GATEWAY_TOKEN_ENV,
    }
)
DEFAULT_AUTH_MODE_BY_PROVIDER = {
    provider: default_auth_mode(provider) for provider in SUPPORTED_PROVIDERS
}
DEFAULT_AUTH_MODE_BY_PROVIDER.update(
    {
        "openclaw": "oauth",
        "oauth_gateway": "gateway",
        "anthropic_gateway": "gateway",
    }
)
REASONING_EFFORTS = frozenset({"inherit", "off", "minimal", "low", "medium", "high", "xhigh", "max"})
HIGH_STAKES_REASONING_EFFORTS = frozenset({"high", "xhigh", "max"})
AUTH_MODES = frozenset({"api_key", "oauth", "gateway", "none"})
API_PROTOCOLS = frozenset({"anthropic_messages", "chat_completions", "custom_gateway", "gemini", "gemini_generate_content", "ollama_openai_compatible", "openai_chat_completions", "openai_codex_responses", "openai_compatible", "openai_responses", "responses"})
THINKING_MODES = frozenset({"auto", "enabled", "disabled"})
CREDENTIAL_SOURCES = frozenset({"api_key_env", "oauth_token", "gateway_token", "none"})


class ModelUseCase(StrEnum):
    PRIMARY_CONTROL = "primary_control"
    LOW_RISK_EXTRACTION = "low_risk_extraction"
    LOW_RISK_SUMMARIZATION = "low_risk_summarization"
    PHYSICS_DECISION = "physics_decision"
    FINAL_RUN_APPROVAL = "final_run_approval"


class AuthMode(StrEnum):
    API_KEY = "api_key"
    OAUTH = "oauth"
    GATEWAY = "gateway"
    NONE = "none"


class ModelPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AgentsSdkModelSpec:
    provider: str
    provider_id: str
    model: str
    api_protocol: str
    base_url: str
    api_key_env: str
    auth_mode: str
    auth_refresh_command: str | None
    reasoning_effort: str
    thinking_mode: str
    structured_outputs: bool
    streaming: bool
    tool_choice_support: bool
    provider_session_support: bool
    credential_source: str


@dataclass(frozen=True, slots=True)
class ModelProviderConfig:
    provider: str
    provider_id: str
    model: str
    api_protocol: str
    reasoning_effort: str
    thinking_mode: str
    base_url: str
    use_case: ModelUseCase
    structured_outputs: bool
    streaming: bool
    tool_choice_support: bool
    provider_session_support: bool
    api_key_env: str
    auth_mode: str
    auth_refresh_command: str | None
    credential_source: str

    @classmethod
    def from_mapping(cls, value: JsonMap) -> ModelProviderConfig:
        mapping = as_mapping(value, "model_provider")
        provider = _normalize_provider_id(_provider_id_from_mapping(mapping))
        base_url = str_field(mapping, "base_url")
        normalized_base_url = normalize_provider_base_url(provider, base_url)

        use_case = _parse_use_case(_optional_str(mapping, "use_case", ModelUseCase.PRIMARY_CONTROL.value))
        model = _optional_str(mapping, "model", PRIMARY_MODEL)
        reasoning_effort = _optional_str(mapping, "reasoning_effort", PRIMARY_REASONING)
        auth_mode = _parse_auth_mode(
            _optional_str(mapping, "auth_mode", DEFAULT_AUTH_MODE_BY_PROVIDER[provider])
        )
        config = cls(
            provider=provider,
            provider_id=provider,
            model=model,
            api_protocol=_parse_api_protocol(_optional_str(mapping, "api_protocol", _default_api_protocol(provider))),
            reasoning_effort=reasoning_effort,
            thinking_mode=_parse_thinking_mode(_optional_str(mapping, "thinking_mode", "auto")),
            base_url=normalized_base_url,
            use_case=use_case,
            structured_outputs=_optional_bool(mapping, "structured_outputs", True),
            streaming=_optional_bool(mapping, "streaming", False),
            tool_choice_support=_optional_bool(mapping, "tool_choice_support", True),
            provider_session_support=_optional_bool(mapping, "provider_session_support", True),
            api_key_env=_optional_str(mapping, "api_key_env", DEFAULT_API_KEY_ENV_BY_PROVIDER[provider]),
            auth_mode=auth_mode,
            auth_refresh_command=_optional_str_or_none(mapping, "auth_refresh_command"),
            credential_source=_parse_credential_source(
                _optional_str(mapping, "credential_source", _default_credential_source(auth_mode))
            ),
        )
        enforce_model_policy(config)
        return config

    def to_agents_sdk_model_spec(self) -> AgentsSdkModelSpec:
        return AgentsSdkModelSpec(
            provider=self.provider,
            provider_id=self.provider_id,
            model=self.model,
            api_protocol=self.api_protocol,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            auth_mode=self.auth_mode,
            auth_refresh_command=self.auth_refresh_command,
            reasoning_effort=self.reasoning_effort,
            thinking_mode=self.thinking_mode,
            structured_outputs=self.structured_outputs,
            streaming=self.streaming,
            tool_choice_support=self.tool_choice_support,
            provider_session_support=self.provider_session_support,
            credential_source=self.credential_source,
        )


def _optional_str(mapping: JsonMap, field: str, default: str) -> str:
    value = mapping.get(field, default)
    if not isinstance(value, str) or not value:
        raise ModelPolicyError(f"{field}_must_be_non_empty_string")
    return value


def _optional_str_or_none(mapping: JsonMap, field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ModelPolicyError(f"{field}_must_be_non_empty_string")
    return value


def _optional_bool(mapping: JsonMap, field: str, default: bool) -> bool:
    value = mapping.get(field, default)
    if not isinstance(value, bool):
        raise ModelPolicyError(f"{field}_must_be_bool")
    return value


def _parse_use_case(value: str) -> ModelUseCase:
    try:
        return ModelUseCase(value)
    except ValueError as exc:
        allowed = ",".join(item.value for item in ModelUseCase)
        raise ModelPolicyError(f"unknown_use_case={value}; allowed={allowed}") from exc


def _parse_auth_mode(value: str) -> str:
    if value not in AUTH_MODES:
        allowed = ",".join(sorted(AUTH_MODES))
        raise ModelPolicyError(f"invalid_auth_mode={value}; allowed={allowed}")
    return value


def _parse_api_protocol(value: str) -> str:
    if value not in API_PROTOCOLS:
        allowed = ",".join(sorted(API_PROTOCOLS))
        raise ModelPolicyError(f"invalid_api_protocol={value}; allowed={allowed}")
    return value


def _parse_thinking_mode(value: str) -> str:
    if value not in THINKING_MODES:
        allowed = ",".join(sorted(THINKING_MODES))
        raise ModelPolicyError(f"invalid_thinking_mode={value}; allowed={allowed}")
    return value


def _parse_credential_source(value: str) -> str:
    if value not in CREDENTIAL_SOURCES:
        allowed = ",".join(sorted(CREDENTIAL_SOURCES))
        raise ModelPolicyError(f"invalid_credential_source={value}; allowed={allowed}")
    return value


def _provider_id_from_mapping(mapping: JsonMap) -> str:
    provider_id = _optional_str_or_none(mapping, "provider_id")
    if provider_id is not None:
        return provider_id
    return str_field(mapping, "provider")


def _default_api_protocol(provider: str) -> str:
    if provider in {"openai", "openai-codex"}:
        return "responses"
    if provider == "anthropic":
        return "anthropic_messages"
    if provider in {"google-gemini-cli", "google-antigravity"}:
        return "gemini"
    return "openai_compatible"


def _default_credential_source(auth_mode: str) -> str:
    if auth_mode == AuthMode.API_KEY:
        return "api_key_env"
    if auth_mode == AuthMode.OAUTH:
        return "oauth_token"
    if auth_mode == AuthMode.NONE:
        return "none"
    return "gateway_token"


def _normalize_provider_id(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        allowed = ",".join(sorted(SUPPORTED_PROVIDERS))
        raise ProviderConfigPolicyError(f"ProviderConfigPolicyError: unsupported_model_provider={provider}; allowed={allowed}")
    return normalized


def normalize_provider_base_url(provider: str, base_url: str) -> str:
    if provider == "openclaw":
        if not is_allowed_openclaw_base_url(base_url):
            raise ProviderConfigPolicyError("ProviderConfigPolicyError: openclaw_base_url_not_allowed")
        return normalize_openclaw_base_url(base_url)
    normalized = _normalize_generic_base_url(base_url)
    if not normalized:
        raise ProviderConfigPolicyError("ProviderConfigPolicyError: model_provider_base_url_invalid")
    return normalized


def _normalize_generic_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = parsed.hostname.lower()
    if port is not None:
        netloc = f"{netloc}:{port}"
    normalized_path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), netloc, normalized_path, "", "", ""))


def enforce_model_policy(config: ModelProviderConfig) -> None:
    if config.reasoning_effort not in REASONING_EFFORTS:
        allowed = ",".join(sorted(REASONING_EFFORTS))
        raise ModelPolicyError(f"reasoning_effort_invalid={config.reasoning_effort}; allowed={allowed}")
    if config.use_case in {
        ModelUseCase.PRIMARY_CONTROL,
        ModelUseCase.PHYSICS_DECISION,
        ModelUseCase.FINAL_RUN_APPROVAL,
    } and config.reasoning_effort not in HIGH_STAKES_REASONING_EFFORTS:
        raise ModelPolicyError("high_stakes_model_requires_high_reasoning")
