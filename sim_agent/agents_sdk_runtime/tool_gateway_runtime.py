from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from sim_agent.agent_harness.tools import (
    RuntimeToolResult,
    ToolRegistry,
    default_tool_registry,
)
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .agent_loop import AgentLoop, AgentLoopResult, AsaAgentSession, ModelSelectedToolCall, StaticToolChoiceModel
from .tool_gateway_policy import DEFAULT_TOOL_GATEWAY_POLICY, ToolGatewayPolicy
from .tool_gateway_runtime_constants import TOOL_GATEWAY_RUNTIME_LEDGER_NAME
from .tool_gateway_sessions import write_tool_gateway_sessions
from .types import RuntimeTraceEvent


@dataclass(frozen=True, slots=True)
class ToolGatewayRuntimeResult:
    run_id: str
    session_id: str
    status: str
    provider: str
    model: str
    auth_mode: str
    gateway_policy_id: str
    gateway_mode: str
    gateway_request_id: str | None
    attached_tools: tuple[str, ...]
    tool_results: tuple[RuntimeToolResult, ...]
    session_files: tuple[str, ...]
    blockers: tuple[str, ...]
    final_output: str
    agent_loop_status: str = ""
    agent_session_agent_id: str = "orchestrator"
    model_selected_tools: tuple[str, ...] = ()
    model_visible_tools: tuple[JsonMap, ...] = ()
    loop_trace: tuple[RuntimeTraceEvent, ...] = ()


class ToolGatewayRuntimeError(RuntimeError):
    pass


def run_agents_sdk_tool_gateway_runtime(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
    *,
    api_key: str | None = None,
    registry: ToolRegistry | None = None,
    policy: ToolGatewayPolicy = DEFAULT_TOOL_GATEWAY_POLICY,
) -> ToolGatewayRuntimeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_registry = registry or default_tool_registry()
    run_id = f"agents-sdk-tool-gateway-{_request_id(payload)}"
    session_id = f"tool-gateway-{_request_id(payload)}"
    attached_tools = _executable_tool_names(active_registry)
    token = api_key or _env_token(endpoint)
    if _credentials_required(endpoint) and token is None:
        return _blocked_result(
            endpoint,
            run_id,
            session_id,
            attached_tools,
            "missing_gateway_credentials",
            policy,
        )
    if endpoint.provider != policy.provider:
        return _blocked_result(
            endpoint,
            run_id,
            session_id,
            attached_tools,
            "live_gateway_tool_dispatch_not_configured",
            policy,
        )
    missing_tools = _missing_plan_tools(active_registry, policy)
    if missing_tools:
        return _blocked_result(
            endpoint,
            run_id,
            session_id,
            attached_tools,
            f"missing_runtime_tools={','.join(missing_tools)}",
            policy,
        )

    loop_result = _run_agent_loop_policy_plan(
        active_registry,
        output_dir,
        run_id,
        session_id,
        endpoint,
        payload,
        policy,
    )
    tool_results = loop_result.tool_results
    session_files = write_tool_gateway_sessions(output_dir, run_id, session_id, tool_results, policy)
    blockers = tuple(result.blocker for result in tool_results if result.blocker is not None)
    status = "succeeded" if not blockers and all(result.status == "succeeded" for result in tool_results) else "failed"
    final_output = "gateway_tool_dispatch_ready" if status == "succeeded" else "gateway_tool_dispatch_failed"
    return ToolGatewayRuntimeResult(
        run_id=run_id,
        session_id=session_id,
        status=status,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        gateway_policy_id=policy.policy_id,
        gateway_mode=policy.mode,
        gateway_request_id=policy.gateway_request_id,
        attached_tools=attached_tools,
        tool_results=tool_results,
        session_files=session_files,
        blockers=blockers,
        final_output=final_output,
        agent_loop_status=loop_result.status,
        agent_session_agent_id=loop_result.session.agent_id,
        model_selected_tools=tuple(selected.tool_name for selected in loop_result.selected_tools),
        model_visible_tools=loop_result.tool_schemas,
        loop_trace=loop_result.trace,
    )


def write_tool_gateway_runtime_ledger(output_dir: Path, result: ToolGatewayRuntimeResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / TOOL_GATEWAY_RUNTIME_LEDGER_NAME
    path.write_text(json.dumps(tool_gateway_runtime_payload(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def tool_gateway_runtime_payload(result: ToolGatewayRuntimeResult) -> JsonMap:
    return {
        "ledger_version": "tool_gateway_runtime_v1",
        "run_id": result.run_id,
        "session_id": result.session_id,
        "status": result.status,
        "provider": result.provider,
        "model": result.model,
        "auth_mode": result.auth_mode,
        "gateway_policy_id": result.gateway_policy_id,
        "gateway_mode": result.gateway_mode,
        "gateway_request_id": result.gateway_request_id or "",
        "attached_tools": list(result.attached_tools),
        "tool_results": [_tool_result_payload(tool_result) for tool_result in result.tool_results],
        "session_files": list(result.session_files),
        "hard_blockers": list(result.blockers),
        "final_output": result.final_output,
        "agent_loop_status": result.agent_loop_status,
        "agent_session_agent_id": result.agent_session_agent_id,
        "model_selected_tools": list(result.model_selected_tools),
        "model_visible_tools": list(result.model_visible_tools),
        "loop_trace": [asdict(event) for event in result.loop_trace],
    }


def _tool_result_payload(result: RuntimeToolResult) -> JsonMap:
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "output": result.output,
        "artifact_ref": result.artifact_ref,
        "blocker": result.blocker or "",
    }


def _run_agent_loop_policy_plan(
    registry: ToolRegistry,
    output_dir: Path,
    run_id: str,
    session_id: str,
    endpoint: ModelProviderConfig,
    payload: JsonMap,
    policy: ToolGatewayPolicy,
) -> AgentLoopResult:
    session = AsaAgentSession(
        run_id=run_id,
        session_id=session_id,
        agent_id="orchestrator",
        user_goal=_user_goal(payload),
        endpoint=endpoint,
        output_dir=output_dir,
        registry=registry,
    )
    model = StaticToolChoiceModel(
        tuple(ModelSelectedToolCall(tool_name, arguments) for tool_name, arguments in policy.plan),
        model_id=f"{endpoint.provider}/{endpoint.model}",
    )
    return AgentLoop(session, model).run()


def _blocked_result(
    endpoint: ModelProviderConfig,
    run_id: str,
    session_id: str,
    attached_tools: tuple[str, ...],
    blocker: str,
    policy: ToolGatewayPolicy,
) -> ToolGatewayRuntimeResult:
    return ToolGatewayRuntimeResult(
        run_id=run_id,
        session_id=session_id,
        status="blocked",
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        gateway_policy_id=policy.policy_id,
        gateway_mode=policy.mode,
        gateway_request_id=None,
        attached_tools=attached_tools,
        tool_results=(),
        session_files=(),
        blockers=(blocker,),
        final_output="blocked",
    )


def _missing_plan_tools(registry: ToolRegistry, policy: ToolGatewayPolicy) -> tuple[str, ...]:
    return tuple(
        tool_name
        for tool_name, _arguments in policy.plan
        if tool_name not in registry.tool_names
    )


def _executable_tool_names(registry: ToolRegistry) -> tuple[str, ...]:
    return tuple(tool.name for tool in registry.tools if tool.executable)


def _credentials_required(endpoint: ModelProviderConfig) -> bool:
    return endpoint.auth_mode in {"api_key", "oauth", "gateway"}


def _env_token(endpoint: ModelProviderConfig) -> str | None:
    value = os.environ.get(endpoint.api_key_env)
    if value:
        return value
    return None


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def _user_goal(payload: JsonMap) -> str:
    value = payload.get("user_goal")
    if isinstance(value, str) and value:
        return value
    return f"Run atomistic simulation request {_request_id(payload)}"
