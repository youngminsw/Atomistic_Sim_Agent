from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never
from urllib.parse import quote

from sim_agent.agents_sdk_runtime.gateway_client_http import gateway_post_json, gateway_url
from sim_agent.agents_sdk_runtime.gateway_client_types import GatewayClientSmokeError
from sim_agent.agents_sdk_runtime.provider_transport_parsers import (
    anthropic_final_text,
    gemini_final_text,
    openai_chat_final_text,
    openai_responses_final_text,
)
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .compaction_provider_auth import provider_headers, provider_token
from .compaction_redaction import redact_secret_text
from .compaction_semantic import (
    SemanticSummaryRequest,
    SemanticSummaryResult,
    SemanticSummaryUnavailable,
    load_compaction_prompt,
    render_provider_compaction_summary,
)


CODEX_BASE_URL: Final = "https://chatgpt.com/backend-api"


class SemanticProviderProtocol(StrEnum):
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_CODEX_RESPONSES = "openai_codex_responses"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"
    OLLAMA_OPENAI_COMPATIBLE = "ollama_openai_compatible"
    CUSTOM_GATEWAY = "custom_gateway"


@dataclass(frozen=True, slots=True)
class ProviderSemanticCompletion:
    endpoint: ModelProviderConfig
    protocol: SemanticProviderProtocol
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class ProviderSemanticSummarizer:
    endpoint: ModelProviderConfig
    api_key: str | None = None
    timeout_s: float = 60.0

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        protocol = _semantic_protocol(self.endpoint)
        completion = ProviderSemanticCompletion(self.endpoint, protocol, request.system_prompt, request.prompt)
        summary = self._complete(completion)
        short_completion = ProviderSemanticCompletion(
            self.endpoint,
            protocol,
            request.system_prompt,
            _short_summary_prompt(summary),
        )
        short_summary = self._complete(short_completion)
        return SemanticSummaryResult(
            summary=summary,
            short_summary=short_summary,
            preserve_data=_preserve_data(completion, request, summary),
        )

    def _complete(self, completion: ProviderSemanticCompletion) -> str:
        token = provider_token(self.endpoint, self.api_key)
        if _auth_required(self.endpoint) and not token:
            raise SemanticSummaryUnavailable("semantic_summary_auth_missing")
        try:
            _status, response = gateway_post_json(
                _semantic_url(self.endpoint, completion.protocol),
                _semantic_payload(completion),
                token,
                self.timeout_s,
                provider_headers(completion.protocol.value, token),
            )
        except GatewayClientSmokeError as exc:
            raise SemanticSummaryUnavailable(str(exc)) from exc
        text = _final_text(completion.protocol, response).strip()
        if not text:
            raise SemanticSummaryUnavailable("empty_semantic_summary")
        return text


def _short_summary_prompt(summary: str) -> str:
    safe_summary = redact_secret_text(summary.strip())
    return "\n\n".join(
        (
            load_compaction_prompt("compaction-short-summary"),
            "<summary>",
            safe_summary,
            "</summary>",
        )
    )


def _auth_required(endpoint: ModelProviderConfig) -> bool:
    return endpoint.auth_mode != "none" and endpoint.credential_source != "none"


def _semantic_payload(completion: ProviderSemanticCompletion) -> JsonMap:
    match completion.protocol:
        case SemanticProviderProtocol.OPENAI_RESPONSES | SemanticProviderProtocol.CUSTOM_GATEWAY:
            return _openai_responses_payload(completion, store=None)
        case SemanticProviderProtocol.OPENAI_CODEX_RESPONSES:
            return _openai_responses_payload(completion, store=False)
        case SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS | SemanticProviderProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return {
                "model": completion.endpoint.model,
                "messages": [
                    {"role": "system", "content": completion.system_prompt},
                    {"role": "user", "content": completion.user_prompt},
                ],
                "temperature": 0,
            }
        case SemanticProviderProtocol.ANTHROPIC_MESSAGES:
            return {
                "model": completion.endpoint.model,
                "max_tokens": 4096,
                "system": completion.system_prompt,
                "messages": [{"role": "user", "content": completion.user_prompt}],
            }
        case SemanticProviderProtocol.GEMINI_GENERATE_CONTENT:
            return {
                "contents": [{"role": "user", "parts": [{"text": completion.user_prompt}]}],
                "systemInstruction": {"parts": [{"text": completion.system_prompt}]},
                "generationConfig": {"temperature": 0},
            }
        case unreachable:
            assert_never(unreachable)


def _openai_responses_payload(completion: ProviderSemanticCompletion, *, store: bool | None) -> JsonMap:
    payload: JsonMap = {
        "model": completion.endpoint.model,
        "instructions": completion.system_prompt,
        "input": [{"role": "user", "content": completion.user_prompt}],
        "reasoning": {"effort": completion.endpoint.reasoning_effort},
    }
    if store is not None:
        payload["store"] = store
    return payload


def _semantic_url(endpoint: ModelProviderConfig, protocol: SemanticProviderProtocol) -> str:
    match protocol:
        case SemanticProviderProtocol.OPENAI_RESPONSES:
            return gateway_url(endpoint.base_url, "/v1/responses")
        case SemanticProviderProtocol.OPENAI_CODEX_RESPONSES:
            return _codex_responses_url(endpoint.base_url)
        case SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS | SemanticProviderProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return gateway_url(endpoint.base_url, "/v1/chat/completions")
        case SemanticProviderProtocol.ANTHROPIC_MESSAGES:
            return gateway_url(endpoint.base_url, "/v1/messages")
        case SemanticProviderProtocol.GEMINI_GENERATE_CONTENT:
            return _gemini_generate_content_url(endpoint.base_url, endpoint.model)
        case SemanticProviderProtocol.CUSTOM_GATEWAY:
            return gateway_url(endpoint.base_url, "/v1/agent/responses")
        case unreachable:
            assert_never(unreachable)


def _semantic_protocol(endpoint: ModelProviderConfig) -> SemanticProviderProtocol:
    value = endpoint.api_protocol.strip().lower()
    match value:
        case "openai_responses" | "responses":
            if endpoint.provider == "openai-codex":
                return SemanticProviderProtocol.OPENAI_CODEX_RESPONSES
            return SemanticProviderProtocol.OPENAI_RESPONSES
        case "openai_codex_responses":
            return SemanticProviderProtocol.OPENAI_CODEX_RESPONSES
        case "openai_chat_completions" | "chat_completions":
            return SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS
        case "anthropic_messages":
            return SemanticProviderProtocol.ANTHROPIC_MESSAGES
        case "gemini_generate_content" | "gemini":
            return SemanticProviderProtocol.GEMINI_GENERATE_CONTENT
        case "ollama_openai_compatible":
            return SemanticProviderProtocol.OLLAMA_OPENAI_COMPATIBLE
        case "custom_gateway":
            return SemanticProviderProtocol.CUSTOM_GATEWAY
        case "openai_compatible":
            if endpoint.provider in {"oauth_gateway", "openclaw"}:
                return SemanticProviderProtocol.OPENAI_RESPONSES
            if endpoint.provider in {"ollama", "lm-studio", "vllm"}:
                return SemanticProviderProtocol.OLLAMA_OPENAI_COMPATIBLE
            return SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS
        case _:
            return SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS


def _preserve_data(
    completion: ProviderSemanticCompletion,
    request: SemanticSummaryRequest,
    summary: str,
) -> JsonMap:
    if completion.protocol not in {
        SemanticProviderProtocol.OPENAI_RESPONSES,
        SemanticProviderProtocol.OPENAI_CODEX_RESPONSES,
        SemanticProviderProtocol.CUSTOM_GATEWAY,
    }:
        return {}
    return {
        "openaiRemoteCompaction": {
            "provider": completion.endpoint.provider,
            "model": completion.endpoint.model,
            "agentId": request.agent_id,
            "compactId": request.compact_id,
            "replacementHistory": [{"type": "message", "role": "user", "content": render_provider_compaction_summary(summary)}],
            "compactionItem": {"type": "compaction_summary", "summary": summary},
        }
    }


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


def _final_text(protocol: SemanticProviderProtocol, response: JsonMap) -> str:
    match protocol:
        case SemanticProviderProtocol.OPENAI_RESPONSES | SemanticProviderProtocol.OPENAI_CODEX_RESPONSES | SemanticProviderProtocol.CUSTOM_GATEWAY:
            return openai_responses_final_text(response)
        case SemanticProviderProtocol.OPENAI_CHAT_COMPLETIONS | SemanticProviderProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return openai_chat_final_text(response)
        case SemanticProviderProtocol.ANTHROPIC_MESSAGES:
            return anthropic_final_text(response)
        case SemanticProviderProtocol.GEMINI_GENERATE_CONTENT:
            return gemini_final_text(response)
        case unreachable:
            assert_never(unreachable)
