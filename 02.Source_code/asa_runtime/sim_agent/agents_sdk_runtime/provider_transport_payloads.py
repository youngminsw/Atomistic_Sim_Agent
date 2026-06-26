from __future__ import annotations

from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession
from .context_assembler import ProviderPromptContext, assemble_provider_context

SUBAGENT_PRESET_ENUM = ["planner", "architect", "critic", "executor", "verifier"]
SUBAGENT_CONTROL_ACTION_ENUM = ["list", "progress", "await", "cancel", "pause", "resume", "steer", "restart"]


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
                    "preset": {"type": "string", "enum": SUBAGENT_PRESET_ENUM},
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
                    "preset": {"type": "string", "enum": SUBAGENT_PRESET_ENUM},
                    "subagent_id": {"type": "string"},
                },
                ("caller_agent", "preset", "subagent_id"),
            )
        case "subagent_control":
            return _object_schema(
                {
                    "action": {"type": "string", "enum": SUBAGENT_CONTROL_ACTION_ENUM},
                    "caller_agent": {"type": "string"},
                    "preset": {"type": "string", "enum": SUBAGENT_PRESET_ENUM},
                    "subagent_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                ("action", "caller_agent"),
            )
        case _:
            return _object_schema({}, ())


def _object_schema(properties: JsonMap, required: tuple[str, ...]) -> JsonMap:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": True}
