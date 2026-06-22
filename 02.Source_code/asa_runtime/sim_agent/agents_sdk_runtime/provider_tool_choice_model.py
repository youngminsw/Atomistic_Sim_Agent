from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap
from sim_agent.ui.model_auth import access_token_for_provider

from .agent_loop import AsaAgentSession, ModelSelectedToolCall, ModelToolChoiceBlocked
from .gateway_client_http import gateway_post_json
from .gateway_client_types import GatewayClientSmokeError
from .provider_transport import ProviderApiProtocol, provider_transport_request, transport_tool_calls


DEFAULT_PROVIDER_MODEL_ID: Final = "provider-tool-choice-model"


@dataclass(frozen=True, slots=True)
class ProviderToolChoiceModel:
    api_key: str | None = None
    timeout_s: float = 60.0
    retry_count: int = 1
    retry_backoff_s: float = 1.0
    model_id: str = DEFAULT_PROVIDER_MODEL_ID

    def model_id_for_session(self, session: AsaAgentSession) -> str:
        if self.model_id != DEFAULT_PROVIDER_MODEL_ID:
            return self.model_id
        return provider_tool_choice_model_id(session)

    def choose_tools(
        self,
        session: AsaAgentSession,
        tool_schemas: Sequence[JsonMap],
    ) -> tuple[ModelSelectedToolCall, ...]:
        model_visible_tools = tuple(tool_schemas)
        allowed_tools, _unsafe_tools = _tool_policy_sets(model_visible_tools)
        safe_tool_schemas = tuple(schema for schema in model_visible_tools if schema.get("name") in allowed_tools)
        request = provider_transport_request(session, safe_tool_schemas)
        token = _token(session, self.api_key)
        response = self._post_with_retry(request.url, request.payload, token, request.protocol)
        return _parse_selected_tools(response, model_visible_tools, request.protocol)

    def _post_with_retry(self, url: str, payload: JsonMap, token: str | None, protocol: ProviderApiProtocol) -> JsonMap:
        for attempt in range(max(0, self.retry_count) + 1):
            try:
                _status, response = gateway_post_json(
                    url,
                    payload,
                    token,
                    self.timeout_s,
                    _provider_headers(protocol, token),
                )
                return response
            except GatewayClientSmokeError as exc:
                if str(exc) != "endpoint_unreachable" or attempt >= max(0, self.retry_count):
                    raise ModelToolChoiceBlocked(str(exc)) from exc
                if self.retry_backoff_s > 0:
                    time.sleep(self.retry_backoff_s)
        raise ModelToolChoiceBlocked("endpoint_unreachable")


def provider_tool_choice_model_id(session: AsaAgentSession) -> str:
    return f"{session.endpoint.provider}/{session.endpoint.model}"


def _request_payload(session: AsaAgentSession, tool_schemas: tuple[JsonMap, ...]) -> JsonMap:
    allowed_tools, _unsafe_tools = _tool_policy_sets(tool_schemas)
    return {
        "model": session.endpoint.model,
        "instructions": (
            "You are the ASA runtime tool selector. Select exactly one safe executable tool. "
            "For a smoke/evidence turn, prefer artifact_write and provide relative_path plus content."
        ),
        "input": [{"role": "user", "content": session.user_goal}],
        "tools": [_provider_tool_schema(schema) for schema in tool_schemas if schema.get("name") in allowed_tools],
        "tool_choice": "required",
        "reasoning": {"effort": session.endpoint.reasoning_effort},
        "metadata": {
            "run_id": session.run_id,
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "provider": session.endpoint.provider,
            "use_case": session.endpoint.use_case.value,
        },
    }


def _token(session: AsaAgentSession, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    value = os.environ.get(session.endpoint.api_key_env)
    if value:
        return value
    return access_token_for_provider(session.endpoint.provider)


def _provider_headers(protocol: ProviderApiProtocol, token: str | None) -> dict[str, str]:
    if protocol is not ProviderApiProtocol.OPENAI_CODEX_RESPONSES:
        return {}
    headers = {
        "OpenAI-Beta": "responses=experimental",
        "originator": "pi",
    }
    account_id = _codex_account_id(token)
    if account_id:
        headers["chatgpt-account-id"] = account_id
    return headers


def _codex_account_id(token: str | None) -> str:
    if not token:
        return ""
    parts = token.split(".")
    if len(parts) != 3:
        return ""
    try:
        payload_raw = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload_raw.encode("utf-8"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    auth = payload.get("https://api.openai.com/auth")
    if not isinstance(auth, dict):
        return ""
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) else ""


def _parse_selected_tools(
    response: JsonMap,
    tool_schemas: tuple[JsonMap, ...],
    protocol: ProviderApiProtocol = ProviderApiProtocol.OPENAI_RESPONSES,
) -> tuple[ModelSelectedToolCall, ...]:
    allowed, unsafe = _tool_policy_sets(tool_schemas)
    calls = tuple(transport_tool_calls(protocol, response))
    if not calls:
        return ()
    selected: list[ModelSelectedToolCall] = []
    for raw_call in calls:
        name, arguments = _tool_name_and_arguments(raw_call)
        if name not in allowed and name not in unsafe:
            raise ModelToolChoiceBlocked("unknown_model_tool_selected")
        if name in unsafe:
            raise ModelToolChoiceBlocked("unsafe_model_tool_selected")
        selected.append(ModelSelectedToolCall(name, arguments))
    return tuple(selected)


def _tool_policy_sets(tool_schemas: tuple[JsonMap, ...]) -> tuple[frozenset[str], frozenset[str]]:
    allowed: set[str] = set()
    unsafe: set[str] = set()
    for schema in tool_schemas:
        name = schema.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name != "bash_process" and schema.get("executable") is True and schema.get("approval_required") is False:
            allowed.add(name)
        else:
            unsafe.add(name)
    return frozenset(allowed), frozenset(unsafe)


def _provider_tool_schema(schema: JsonMap) -> JsonMap:
    name = schema.get("name")
    if not isinstance(name, str) or not name:
        name = "unknown_tool"
    return {
        "type": "function",
        "name": name,
        "description": _tool_description(schema),
        "parameters": _tool_parameters(name),
    }


def _tool_description(schema: JsonMap) -> str:
    name = schema.get("name")
    boundary = schema.get("boundary")
    policy = schema.get("policy_summary")
    parts = [value for value in (name, boundary, policy) if isinstance(value, str) and value]
    return " / ".join(parts) if parts else "ASA runtime tool"


def _tool_parameters(name: str) -> JsonMap:
    match name:
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
        case _:
            return _object_schema({}, ())


def _object_schema(properties: JsonMap, required: tuple[str, ...]) -> JsonMap:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": True,
    }


def _iter_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
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


def _tool_name_and_arguments(raw_call: JsonMap) -> tuple[str, JsonMap]:
    function = raw_call.get("function")
    source = function if isinstance(function, dict) else raw_call
    name = source.get("name") or raw_call.get("name")
    if not isinstance(name, str) or not name:
        raise ModelToolChoiceBlocked("malformed_model_tool_call")
    arguments = source.get("arguments", raw_call.get("arguments", {}))
    return name, _arguments_map(arguments)


def _arguments_map(arguments) -> JsonMap:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ModelToolChoiceBlocked("malformed_model_tool_call") from exc
        if isinstance(decoded, dict):
            return decoded
    raise ModelToolChoiceBlocked("malformed_model_tool_call")
