from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote

from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession
from .context_assembler import assemble_provider_context
from .gateway_client_http import gateway_url

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


def provider_transport_request(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> ProviderHttpRequest:
    protocol = api_protocol_for_session(session)
    match protocol:
        case ProviderApiProtocol.OPENAI_RESPONSES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/responses"),
                payload=_openai_responses_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.OPENAI_CODEX_RESPONSES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=_codex_responses_url(session.endpoint.base_url),
                payload=_openai_codex_responses_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS | ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/chat/completions"),
                payload=_openai_chat_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.ANTHROPIC_MESSAGES:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/messages"),
                payload=_anthropic_messages_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.GEMINI_GENERATE_CONTENT:
            return ProviderHttpRequest(
                protocol=protocol,
                url=_gemini_generate_content_url(session.endpoint.base_url, session.endpoint.model),
                payload=_gemini_generate_content_payload(session, tool_schemas),
            )
        case ProviderApiProtocol.CUSTOM_GATEWAY:
            return ProviderHttpRequest(
                protocol=protocol,
                url=gateway_url(session.endpoint.base_url, "/v1/agent/responses"),
                payload=_openai_responses_payload(session, tool_schemas),
            )


def api_protocol_for_session(session: AsaAgentSession) -> ProviderApiProtocol:
    explicit = getattr(session.endpoint, "api_protocol", None)
    if isinstance(explicit, str) and explicit:
        return _parse_protocol(explicit, session.endpoint.provider)
    return _default_protocol_for_provider(session.endpoint.provider)


def transport_tool_calls(protocol: ProviderApiProtocol, response: JsonMap) -> tuple[JsonMap, ...]:
    match protocol:
        case ProviderApiProtocol.OPENAI_RESPONSES | ProviderApiProtocol.OPENAI_CODEX_RESPONSES | ProviderApiProtocol.CUSTOM_GATEWAY:
            return _openai_responses_tool_calls(response)
        case ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS | ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE:
            return _openai_chat_tool_calls(response)
        case ProviderApiProtocol.ANTHROPIC_MESSAGES:
            return _anthropic_tool_calls(response)
        case ProviderApiProtocol.GEMINI_GENERATE_CONTENT:
            return _gemini_tool_calls(response)


def _parse_protocol(value: str, provider: str = "") -> ProviderApiProtocol:
    normalized = value.strip().lower()
    if normalized == "responses":
        if provider.strip().lower() == "openai-codex":
            return ProviderApiProtocol.OPENAI_CODEX_RESPONSES
        return ProviderApiProtocol.OPENAI_RESPONSES
    if normalized in {"chat_completions", "openai_compatible"}:
        normalized_provider = provider.strip().lower()
        if normalized_provider in {"oauth_gateway", "openclaw"}:
            return ProviderApiProtocol.OPENAI_RESPONSES
        if normalized_provider in {"ollama", "lm-studio", "vllm"}:
            return ProviderApiProtocol.OLLAMA_OPENAI_COMPATIBLE
        return ProviderApiProtocol.OPENAI_CHAT_COMPLETIONS
    if normalized == "gemini":
        return ProviderApiProtocol.GEMINI_GENERATE_CONTENT
    try:
        return ProviderApiProtocol(normalized)
    except ValueError:
        return _default_protocol_for_provider(normalized)


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


def _openai_responses_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "instructions": context.instructions,
        "input": context.openai_responses_input(),
        "tools": [_openai_responses_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": "required",
        "reasoning": {"effort": session.endpoint.reasoning_effort},
        "metadata": _metadata(session),
    }


def _openai_codex_responses_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    payload = _openai_responses_payload(session, tool_schemas)
    payload["store"] = False
    payload["stream"] = True
    payload.pop("metadata", None)
    return payload


def _codex_responses_url(base_url: str) -> str:
    raw = base_url.strip() if base_url.strip() else CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _openai_chat_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "messages": context.openai_chat_messages(),
        "tools": [_openai_chat_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": "required",
        "metadata": _metadata(session),
    }


def _anthropic_messages_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "max_tokens": 1024,
        "system": context.instructions,
        "messages": context.anthropic_messages(),
        "tools": [_anthropic_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": {"type": "any"},
        "metadata": _metadata(session),
    }


def _gemini_generate_content_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "contents": context.gemini_contents(),
        "systemInstruction": {"parts": [{"text": context.instructions}]},
        "tools": [{"functionDeclarations": [_gemini_tool_schema(schema) for schema in tool_schemas]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        "generationConfig": {"temperature": 0},
    }


def _metadata(session: AsaAgentSession) -> JsonMap:
    return {
        "run_id": session.run_id,
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "provider": session.endpoint.provider,
        "use_case": session.endpoint.use_case.value,
    }


def _openai_responses_tool_schema(schema: JsonMap) -> JsonMap:
    return {
        "type": "function",
        "name": _tool_name(schema),
        "description": _tool_description(schema),
        "parameters": _tool_parameters(schema),
    }


def _openai_chat_tool_schema(schema: JsonMap) -> JsonMap:
    return {
        "type": "function",
        "function": {
            "name": _tool_name(schema),
            "description": _tool_description(schema),
            "parameters": _tool_parameters(schema),
        },
    }


def _anthropic_tool_schema(schema: JsonMap) -> JsonMap:
    return {
        "name": _tool_name(schema),
        "description": _tool_description(schema),
        "input_schema": _tool_parameters(schema),
    }


def _gemini_tool_schema(schema: JsonMap) -> JsonMap:
    return {
        "name": _tool_name(schema),
        "description": _tool_description(schema),
        "parameters": _tool_parameters(schema),
    }


def _tool_name(schema: JsonMap) -> str:
    name = schema.get("name")
    return name if isinstance(name, str) and name else "unknown_tool"


def _tool_description(schema: JsonMap) -> str:
    name = schema.get("name")
    boundary = schema.get("boundary")
    policy = schema.get("policy_summary")
    parts = [value for value in (name, boundary, policy) if isinstance(value, str) and value]
    return " / ".join(parts) if parts else "ASA runtime tool"


def _tool_parameters(schema: JsonMap) -> JsonMap:
    parameters = schema.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    match _tool_name(schema):
        case "artifact_write":
            return _object_schema(
                {
                    "relative_path": {"type": "string", "description": "Relative artifact path inside this ASA session."},
                    "content": {"type": "string", "description": "Evidence text to write."},
                },
                ("relative_path", "content"),
            )
        case "graphdb_dry_run":
            return _object_schema({"database_name": {"type": "string"}}, ("database_name",))
        case "agent_message":
            return _object_schema(
                {
                    "action": {"type": "string", "enum": ["send", "ack", "read", "reply"]},
                    "from_agent": {"type": "string"},
                    "to_agent": {"type": "string"},
                    "content": {"type": "string"},
                    "message_id": {"type": "string"},
                    "by_agent": {"type": "string"},
                    "thread_id": {"type": "string"},
                },
                ("action",),
            )
        case "handoff_task":
            return _object_schema(
                {
                    "target_agent": {"type": "string"},
                    "task": {"type": "string"},
                    "from_agent": {"type": "string"},
                    "task_id": {"type": "string"},
                    "thread_id": {"type": "string"},
                },
                ("target_agent", "task"),
            )
        case "subagent_task":
            return _object_schema(
                {
                    "caller_agent": {"type": "string"},
                    "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
                    "task": {"type": "string"},
                    "task_id": {"type": "string"},
                    "depth": {"type": "integer"},
                },
                ("caller_agent", "preset", "task"),
            )
        case "subagent_inspect":
            return _object_schema(
                {
                    "caller_agent": {"type": "string"},
                    "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
                    "subagent_id": {"type": "string"},
                },
                ("caller_agent", "preset", "subagent_id"),
            )
        case "subagent_control":
            return _object_schema(
                {
                    "action": {
                        "type": "string",
                        "enum": ["list", "progress", "await", "cancel", "pause", "resume", "steer"],
                    },
                    "caller_agent": {"type": "string"},
                    "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
                    "subagent_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                ("action", "caller_agent"),
            )
        case _:
            return _object_schema({}, ())


def _object_schema(properties: JsonMap, required: tuple[str, ...]) -> JsonMap:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": True,
    }


def _gemini_generate_content_url(base_url: str, model: str) -> str:
    root = base_url.rstrip("/").removesuffix("/v1").removesuffix("/v1beta")
    return f"{root}/v1beta/models/{quote(model, safe='')}:generateContent"


def _openai_responses_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    output = response.get("output")
    if isinstance(output, list):
        calls = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") in {"tool_call", "function_call"}
        ]
        if calls:
            return tuple(calls)
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = [item for item in tool_calls if isinstance(item, dict)]
        if calls:
            return tuple(calls)
    return ()


def _openai_chat_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return ()
    calls: list[JsonMap] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend(item for item in tool_calls if isinstance(item, dict))
    return tuple(calls)


def _anthropic_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    content = response.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(
        {"name": item.get("name"), "arguments": item.get("input", {})}
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_use"
    )


def _gemini_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    calls: list[JsonMap] = []
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                calls.append({"name": function_call.get("name"), "arguments": function_call.get("args", {})})
    return tuple(calls)
