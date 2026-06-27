from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession
from .context_assembler import ProviderPromptContext, assemble_provider_context

TOOL_NAME_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class ProviderToolSchemaError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def openai_responses_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "instructions": context.instructions,
        "input": context.openai_responses_input(),
        "tools": [_openai_responses_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": "auto",
        "reasoning": {"effort": session.endpoint.reasoning_effort},
        "metadata": _metadata(session, context),
    }


def openai_codex_responses_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    payload = openai_responses_payload(session, tool_schemas)
    payload["store"] = False
    payload["stream"] = True
    payload.pop("metadata", None)
    return payload


def openai_chat_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "messages": context.openai_chat_messages(),
        "tools": [_openai_chat_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": "auto",
        "metadata": _metadata(session, context),
    }


def anthropic_messages_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "model": session.endpoint.model,
        "max_tokens": 1024,
        "system": context.instructions,
        "messages": context.anthropic_messages(),
        "tools": [_anthropic_tool_schema(schema) for schema in tool_schemas],
        "tool_choice": {"type": "auto"},
        "metadata": _metadata(session, context),
    }


def gemini_generate_content_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    context = assemble_provider_context(session)
    return {
        "contents": context.gemini_contents(),
        "systemInstruction": {"parts": [{"text": context.instructions}]},
        "tools": [{"functionDeclarations": [_gemini_tool_schema(schema) for schema in tool_schemas]}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0},
    }


def _metadata(session: AsaAgentSession, context: ProviderPromptContext) -> JsonMap:
    payload = {
        "run_id": session.run_id,
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "provider": session.endpoint.provider,
        "use_case": session.endpoint.use_case.value,
    }
    payload.update(_compaction_metadata(session, context))
    return payload


def _compaction_metadata(session: AsaAgentSession, context: ProviderPromptContext) -> JsonMap:
    if not session.compaction_metadata:
        return {
            "raw_message_count": str(session.raw_message_count or len(session.messages)),
            "provider_visible_message_count": str(len(context.messages)),
            "provider_messages_rewritten": "false",
        }
    fields = {
        "compact_id",
        "compact_mode",
        "summary_source",
        "first_kept_message_sequence",
        "summary_cutoff_message_sequence",
        "raw_message_count",
        "provider_visible_message_count",
        "short_summary",
        "provider_cache_invalidated",
        "provider_session_reset",
        "preserve_data_openai_remote",
    }
    payload = {field: str(session.compaction_metadata.get(field, "")) for field in fields}
    payload["provider_visible_message_count"] = str(len(context.messages))
    payload["provider_messages_rewritten"] = "true"
    return payload


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
    if isinstance(name, str) and TOOL_NAME_PATTERN.fullmatch(name):
        return name
    raise ProviderToolSchemaError("malformed_tool_schema:name")


def _tool_description(schema: JsonMap) -> str:
    name = schema.get("name")
    boundary = schema.get("boundary")
    policy = schema.get("policy_summary")
    parts = [value for value in (name, boundary, policy) if isinstance(value, str) and value]
    return " / ".join(parts) if parts else "ASA runtime tool"


def _tool_parameters(schema: JsonMap) -> JsonMap:
    parameters = schema.get("parameters")
    if isinstance(parameters, dict):
        return _object_parameters(parameters)
    input_schema = schema.get("inputSchema")
    if isinstance(input_schema, dict):
        return _object_parameters(input_schema)
    raise ProviderToolSchemaError("malformed_tool_schema:parameters")


def _object_parameters(parameters: JsonMap) -> JsonMap:
    if parameters.get("type") != "object":
        raise ProviderToolSchemaError("malformed_tool_schema:parameters")
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ProviderToolSchemaError("malformed_tool_schema:parameters")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(isinstance(field, str) for field in required):
        raise ProviderToolSchemaError("malformed_tool_schema:parameters")
    return parameters
