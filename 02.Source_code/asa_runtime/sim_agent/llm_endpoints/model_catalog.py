from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from sim_agent.provider_registry import (
    LEGACY_PROVIDER_GATEWAY_BASE_URL,
    LOCAL_GATEWAY_BASE_URL,
    default_api_key_env,
    default_auth_mode,
    provider_by_id,
)


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    company: str
    source_provider: str
    provider: str
    provider_id: str
    model: str
    api_protocol: str
    reasoning_effort: str
    thinking_mode: str
    base_url: str
    auth_mode: str
    api_key_env: str
    streaming: bool
    tool_choice_support: bool
    provider_session_support: bool
    credential_source: str
    role_hint: str

    @property
    def reference(self) -> str:
        return f"{self.source_provider}/{self.model}"

    @property
    def runtime_reference(self) -> str:
        return f"{self.provider}/{self.model}"


def _entry(company: str, provider: str, model: str, reasoning_effort: str, role_hint: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        company=company,
        source_provider=provider,
        provider=provider,
        provider_id=provider,
        model=model,
        api_protocol=_api_protocol(provider),
        reasoning_effort=reasoning_effort,
        thinking_mode=_thinking_mode(reasoning_effort),
        base_url=_base_url(provider),
        auth_mode=default_auth_mode(provider),
        api_key_env=default_api_key_env(provider),
        streaming=True,
        tool_choice_support=True,
        provider_session_support=True,
        credential_source=_credential_source(default_auth_mode(provider)),
        role_hint=role_hint,
    )


def _local(model: str, reasoning_effort: str, role_hint: str) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        company="Local / Custom",
        source_provider="local_gateway",
        provider="local_gateway",
        provider_id="local_gateway",
        model=model,
        api_protocol="custom_gateway",
        reasoning_effort=reasoning_effort,
        thinking_mode=_thinking_mode(reasoning_effort),
        base_url=LOCAL_GATEWAY_BASE_URL,
        auth_mode="none",
        api_key_env="RUNTIME_GATEWAY_TOKEN",
        streaming=True,
        tool_choice_support=True,
        provider_session_support=True,
        credential_source="none",
        role_hint=role_hint,
    )


def _base_url(provider: str) -> str:
    spec = provider_by_id(provider)
    if spec is None:
        return LEGACY_PROVIDER_GATEWAY_BASE_URL
    return spec.default_base_url


def _api_protocol(provider: str) -> str:
    if provider in {"openai", "openai-codex"}:
        return "responses"
    if provider == "anthropic":
        return "anthropic_messages"
    if provider in {"google-gemini-cli", "google-antigravity"}:
        return "gemini"
    if provider == "local_gateway":
        return "custom_gateway"
    return "openai_compatible"


def _thinking_mode(reasoning_effort: str) -> str:
    if reasoning_effort in {"off", "minimal"}:
        return "disabled"
    return "enabled"


def _credential_source(auth_mode: str) -> str:
    if auth_mode == "api_key":
        return "api_key_env"
    if auth_mode == "oauth":
        return "oauth_token"
    if auth_mode == "none":
        return "none"
    return "gateway_token"


MODEL_CATALOG: Final[tuple[ModelCatalogEntry, ...]] = (
    _entry("OpenAI", "openai-codex", "gpt-5.5", "xhigh", "primary-subscription"),
    _entry("OpenAI", "openai-codex", "gpt-5.4", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5.4-mini", "medium", "standard"),
    _entry("OpenAI", "openai-codex", "gpt-5.4-nano", "low", "fast"),
    _entry("OpenAI", "openai-codex", "gpt-5.2", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5.1", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5-codex", "high", "primary-codex"),
    _entry("OpenAI", "openai-codex", "gpt-5.1-codex", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5.1-codex-max", "high", "deep"),
    _entry("OpenAI", "openai-codex", "gpt-5.1-codex-mini", "medium", "standard"),
    _entry("OpenAI", "openai-codex", "gpt-5.2-codex", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5.3-codex", "high", "frontier"),
    _entry("OpenAI", "openai-codex", "gpt-5.3-codex-spark", "low", "fast"),
    _entry("OpenAI", "openai-codex", "codex-auto-review", "high", "review"),
    _entry("OpenAI", "openai", "gpt-5.5", "xhigh", "primary-api"),
    _entry("OpenAI", "openai", "gpt-5.3", "high", "standard-api"),
    _entry("OpenAI", "openai", "gpt-5.3-mini", "medium", "fast-api"),
    _entry("OpenAI", "openai", "gpt-5.3-codex-spark", "low", "fast-api"),
    _entry("Anthropic", "anthropic", "claude-opus-4.6", "high", "deep"),
    _entry("Anthropic", "anthropic", "claude-opus-4.5", "high", "deep"),
    _entry("Anthropic", "anthropic", "claude-sonnet-4.5", "high", "standard"),
    _entry("Anthropic", "anthropic", "claude-haiku-4-5", "low", "fast"),
    _entry("Google", "google-gemini-cli", "gemini-3.1-pro-preview", "high", "frontier"),
    _entry("Google", "google-gemini-cli", "gemini-3-pro-preview", "high", "primary"),
    _entry("Google", "google-gemini-cli", "gemini-3-flash-preview", "medium", "standard"),
    _entry("Google", "google-gemini-cli", "gemini-2.5-pro", "high", "legacy-frontier"),
    _entry("Google", "google-gemini-cli", "gemini-2.5-flash", "low", "fast"),
    _entry("Google", "google-antigravity", "gemini-3-pro-high", "high", "antigravity-primary"),
    _entry("Google", "google-antigravity", "claude-sonnet-4-5", "high", "antigravity-claude"),
    _entry("Google", "google-antigravity", "gpt-oss-120b", "medium", "antigravity-open"),
    _entry("Developer Tools", "github-copilot", "claude-sonnet-4.5", "high", "copilot-standard"),
    _entry("Developer Tools", "github-copilot", "gemini-3-pro-preview", "high", "copilot-google"),
    _entry("Developer Tools", "github-copilot", "gpt-5-codex", "high", "copilot-openai"),
    _entry("Developer Tools", "github-copilot", "gpt-5.3-codex-spark", "low", "copilot-fast"),
    _entry("Developer Tools", "cursor", "claude-4.6-sonnet-medium", "high", "cursor-standard"),
    _entry("Developer Tools", "cursor", "claude-4.5-opus-high", "high", "cursor-deep"),
    _entry("Developer Tools", "cursor", "composer-1.5", "medium", "cursor-composer"),
    _entry("Developer Tools", "cursor", "gpt-5-codex", "high", "cursor-openai"),
    _entry("xAI", "xai", "grok-4", "high", "primary"),
    _entry("xAI", "xai", "grok-3", "high", "standard"),
    _entry("xAI", "xai", "grok-3-fast", "medium", "fast"),
    _entry("xAI", "xai", "grok-3-mini-fast", "low", "spark"),
    _entry("DeepSeek", "deepseek", "deepseek-v4-pro", "high", "primary"),
    _entry("DeepSeek", "deepseek", "deepseek-v4-flash", "medium", "fast"),
    _entry("Moonshot / Kimi", "kimi-code", "kimi-k2.7-code", "high", "coding-plan"),
    _entry("Moonshot / Kimi", "kimi-code", "kimi-k2.5", "high", "coding-plan"),
    _entry("Moonshot / Kimi", "moonshot", "kimi-k2.5", "high", "api-plan"),
    _entry("Alibaba / Qwen", "qwen-portal", "qwen3-coder-plus", "high", "portal-coder"),
    _entry("Alibaba / Qwen", "qwen-portal", "qwen3-max", "high", "portal-frontier"),
    _entry("Alibaba / Qwen", "alibaba-coding-plan", "qwen3-coder-plus", "high", "coding-plan"),
    _entry("Z.AI", "zai", "glm-4.7", "high", "primary"),
    _entry("Z.AI", "zai", "glm-4.6", "high", "standard"),
    _entry("MiniMax", "minimax-code", "minimax-m2", "high", "coding-plan"),
    _entry("Perplexity", "perplexity", "sonar-pro", "medium", "research"),
    _entry("Perplexity", "perplexity", "sonar-reasoning-pro", "high", "research-deep"),
    _entry("Fireworks", "fireworks", "accounts/fireworks/models/kimi-k2-instruct", "high", "api-frontier"),
    _entry("Fireworks", "firepass", "kimi-k2-instruct", "high", "subscription-frontier"),
    _entry("Together", "together", "deepseek-ai/DeepSeek-V3.1", "high", "api-frontier"),
    _entry("Cerebras", "cerebras", "qwen-3-coder-480b", "high", "api-coder"),
    _entry("Hugging Face", "huggingface", "openai/gpt-oss-120b", "high", "router-open"),
    _entry("NVIDIA", "nvidia", "qwen/qwen3-coder-480b-a35b-instruct", "high", "nim-coder"),
    _entry("NanoGPT", "nanogpt", "gpt-5.5", "high", "api-aggregator"),
    _entry("Venice", "venice", "qwen3-coder", "high", "api-coder"),
    _local("gpt-5.5", "high", "local-primary"),
    _local("gpt-5.3", "high", "local-standard"),
    _local("gpt-5.3-codex-spark", "low", "local-fast"),
    _local("qwen3-coder", "high", "local-coder"),
    _local("gpt-oss-120b", "high", "local-open"),
)


def list_model_catalog(provider: str | None = None) -> tuple[ModelCatalogEntry, ...]:
    if provider is None:
        return MODEL_CATALOG
    normalized = provider.strip().lower()
    return tuple(
        entry
        for entry in MODEL_CATALOG
        if entry.provider == normalized or entry.source_provider == normalized or entry.company.lower() == normalized
    )


def find_model_catalog_entry(reference: str) -> ModelCatalogEntry | None:
    normalized = reference.strip().lower()
    for entry in MODEL_CATALOG:
        if normalized in {entry.reference.lower(), entry.runtime_reference.lower()}:
            return entry
    return None


def model_catalog_references(entries: Iterable[ModelCatalogEntry] = MODEL_CATALOG) -> str:
    return ",".join(entry.reference for entry in entries)
