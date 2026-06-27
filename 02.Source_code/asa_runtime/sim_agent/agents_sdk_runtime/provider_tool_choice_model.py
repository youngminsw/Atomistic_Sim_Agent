from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap
from sim_agent.agent_runtime.compaction_policy import ProviderContextCompactionBlocked
from sim_agent.ui.model_auth import access_token_for_provider

from .agent_loop import AsaAgentSession, ModelSelectedToolCall, ModelToolChoiceBlocked, ModelTurnResult
from .context_assembler import assemble_provider_context
from .gateway_client_http import gateway_post_json
from .gateway_client_types import GatewayClientSmokeError
from .provider_transport import (
    ProviderApiProtocol,
    ProviderTransportPolicyError,
    provider_transport_request,
    transport_tool_calls,
    transport_final_text,
)


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
        return self.complete_turn(session, tool_schemas).selected_tools

    def complete_turn(
        self,
        session: AsaAgentSession,
        tool_schemas: Sequence[JsonMap],
    ) -> ModelTurnResult:
        model_visible_tools = tuple(tool_schemas)
        allowed_tools, _unsafe_tools = _tool_policy_sets(model_visible_tools)
        safe_tool_schemas = tuple(schema for schema in model_visible_tools if schema.get("name") in allowed_tools)
        try:
            request = provider_transport_request(session, safe_tool_schemas)
        except (ProviderTransportPolicyError, ProviderContextCompactionBlocked) as exc:
            raise ModelToolChoiceBlocked(str(exc)) from exc
        token = _token(session, self.api_key)
        _write_prompt_manifest(session, request.protocol, request.url, safe_tool_schemas)
        response = self._post_with_retry(request.url, request.payload, token, request.protocol)
        return ModelTurnResult(
            selected_tools=_parse_selected_tools(response, model_visible_tools, request.protocol),
            final_output=transport_final_text(request.protocol, response),
        )

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


def _write_prompt_manifest(
    session: AsaAgentSession,
    protocol: ProviderApiProtocol,
    url: str,
    tool_schemas: tuple[JsonMap, ...],
) -> None:
    context = assemble_provider_context(session)
    payload: JsonMap = {
        "schema_version": "asa_prompt_assembly_manifest_v1",
        "run_id": session.run_id,
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "provider": session.endpoint.provider,
        "model": session.endpoint.model,
        "reasoning_effort": session.endpoint.reasoning_effort,
        "auth_mode": session.endpoint.auth_mode,
        "credential_source": session.endpoint.credential_source,
        "api_protocol": protocol.value,
        "url": url,
        "layer_kinds": list(context.layer_kinds()),
        "layers": context.layers_json(),
        "messages": context.openai_responses_input(),
        "raw_message_count": session.raw_message_count or len(session.messages),
        "provider_visible_message_count": len(context.messages),
        "provider_messages_rewritten": bool(session.compaction_metadata),
        "compaction": dict(session.compaction_metadata),
        "tool_names": [_tool_schema_name(schema) for schema in tool_schemas],
    }
    path = session.output_dir / "prompt_assembly_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tool_schema_name(schema: JsonMap) -> str:
    name = schema.get("name")
    return name if isinstance(name, str) else ""


def _token(session: AsaAgentSession, api_key: str | None) -> str | None:
    credential_source = session.endpoint.credential_source.strip().lower()
    if not credential_source:
        raise ModelToolChoiceBlocked("explicit_credential_source_required")
    if credential_source == "none":
        return None
    if api_key:
        return api_key
    if credential_source == "api_key_env":
        value = os.environ.get(session.endpoint.api_key_env)
        if value:
            return value
        raise ModelToolChoiceBlocked("missing_api_key_env")
    if credential_source == "gateway_token":
        value = os.environ.get(session.endpoint.api_key_env)
        if value:
            return value
        raise ModelToolChoiceBlocked("missing_gateway_token")
    if credential_source == "oauth_token":
        token = access_token_for_provider(session.endpoint.provider)
        if token:
            return token
        raise ModelToolChoiceBlocked("missing_oauth_token")
    raise ModelToolChoiceBlocked(f"unsupported_credential_source={session.endpoint.credential_source}")


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
        if (
            schema.get("family") != "process"
            and schema.get("executable") is True
            and schema.get("approval_required") is False
        ):
            allowed.add(name)
        else:
            unsafe.add(name)
    return frozenset(allowed), frozenset(unsafe)


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
