from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote

from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession
from .gateway_client_http import gateway_url
from .provider_transport_parsers import (
    anthropic_final_text,
    anthropic_tool_calls,
    gemini_final_text,
    gemini_tool_calls,
    openai_chat_final_text,
    openai_chat_tool_calls,
    openai_responses_final_text,
    openai_responses_tool_calls,
)
from .provider_transport_payloads import (
    anthropic_messages_payload,
    gemini_generate_content_payload,
    openai_chat_payload,
    openai_codex_responses_payload,
    openai_responses_payload,
)

CODEX_BASE_URL = "https://chatgpt.com/backend-api"


class ProviderApiProtocol(StrEnum):
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_CODEX_RESPONSES = "openai_codex_responses"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"
    OLLAMA_OPENAI_COMPATIBLE = "ollama_openai_compatible"
    CUSTOM_GATEWAY = "custom_gateway"


@dataclass(frozen=True, slots=True)
class ProviderHttpRequest:
    protocol: ProviderApiProtocol
    url: str
    payload: JsonMap


@dataclass(frozen=True, slots=True)
class ProviderTransportPolicyError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def provider_transport_request(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> ProviderHttpRequest:
    protocol = api_protocol_for_session(session)
    match protocol:
        case ProviderApiProtocol.OPENAI_RESPONSES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/responses"),
                payload=openai_responses_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.OPENAI_CODEX_RESPONSES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=_codex_responses_url(session.endpoint.base_url),
                payload=openai_codex_responses_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS | ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/chat/completions"),
                payload=openai_chat_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.ANTHROPIC_MESSAGES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/messages"),
                payload=anthropic_messages_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.GEMINI_GENERATE_CONTENT:
            return ProviderHttpRequest(
                protocol=protocol,
                url=_gemini_generate_content_url(session.endpoint.base_url, session.endpoint.model),
                payload=gemini_generate_content_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.CUSTOM_GATEWAY:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/agent/responses"),
                payload=openai_responses_payload(session, tool_schemas),
            )


def api_protocol_for_session(session: AsaAgentSession) -> ProviderApiProtocol:
    explicit = getattr(session.endpoint, "api_protocol", None)
    if isinstance(explicit, str) and explicit:
        return _parse_protocol(explicit, session.endpoint.provider)
    return _default_protocol_for_provider(session.endpoint.provider)


def transport_tool_calls(protocol: ProviderApiProtocol, response: JsonMap) -> tuple[JsonMap, ...]:
    match protocol:
        case ProviderApiProtocol.OPENAI_RESPONSES | ProviderApiProtocol.OPENAI_CODEX_RESPONSES | ProviderApiProtocol.CUSTOM_GATEWAY:
            return openai_responses_tool_calls(response)
        case ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS | ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return openai_chat_tool_calls(response)
        case ProviderApiProtocol.ANTHROPIC_MESSAGES:
            return anthropic_tool_calls(response)
        case ProviderApiProtocol.GEMINI_GENERATE_CONTENT:
            return gemini_tool_calls(response)


def transport_final_text(protocol: ProviderApiProtocol, response: JsonMap) -> str:
    match protocol:
        case ProviderApiProtocol.OPENAI_RESPONSES | ProviderApiProtocol.OPENAI_CODEX_RESPONSES | ProviderApiProtocol.CUSTOM_GATEWAY:
            return openai_responses_final_text(response)
        case ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS | ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return openai_chat_final_text(response)
        case ProviderApiProtocol.ANTHROPIC_MESSAGES:
            return anthropic_final_text(response)
        case ProviderApiProtocol.GEMINI_GENERATE_CONTENT:
            return gemini_final_text(response)


def _parse_protocol(value: str, provider: str = "") -> ProviderApiProtocol:
    normalized = value.strip().lower()
    try:
        return ProviderApiProtocol(normalized)
    except ValueError:
        pass
    match normalized:
        case "responses":
            if provider.strip().lower() == "openai-codex":
                return ProviderApiProtocol.OPENAI_CODEX_RESPONSES
            return ProviderApiProtocol.OPENAI_RESPONSES
        case "chat_completions":
            return ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS
        case "openai_compatible":
            normalized_provider = provider.strip().lower()
            if normalized_provider in {"oauth_gateway", "openclaw"}:
                return ProviderApiProtocol.OPENAI_RESPONSES
            if normalized_provider in {"ollama", "lm-studio", "vllm"}:
                return ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE
            return ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS
        case "gemini":
            return ProviderApiProtocol.GEMINI_GENERATE_CONTENT
        case _:
            allowed = ",".join(sorted(protocol.value for protocol in ProviderApiProtocol))
            raise ProviderTransportPolicyError(f"invalid_api_protocol={value}; allowed={allowed}")


def _default_protocol_for_provider(provider: str) -> ProviderApiProtocol:
    normalized = provider.strip().lower()
    if normalized == "openai-codex":
        return ProviderApiProtocol.OPENAI_CODEX_RESPONSES
    if normalized in {"openai", "oauth_gateway", "openclaw"}:
        return ProviderApiProtocol.OPENAI_RESPONSES
    if normalized == "anthropic":
        return ProviderApiProtocol.ANTHROPIC_MESSAGES
    if normalized.startswith("google-"):
        return ProviderApiProtocol.GEMINI_GENERATE_CONTENT
    if normalized in {"ollama", "lm-studio", "vllm"}:
        return ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE
    if normalized in {"local_gateway", "custom_gateway"}:
        return ProviderApiProtocol.CUSTOM_GATEWAY
    return ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS


def _codex_responses_url(base_url: str) -> str:
    raw = base_url.strip() if base_url.strip() else CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _gemini_generate_content_url(base_url: str, model: str) -> str:
    root = base_url.rstrip("/").removesuffix("/v1").removesuffix("/v1beta")
    return f"{root}/v1beta/models/{quote(model, safe='')}:generateContent"
