from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime.roles import AGENT_ROLES
from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run, write_agents_sdk_runtime_ledger
from sim_agent.agents_sdk_runtime.tool_gateway_runtime import (
    run_agents_sdk_tool_gateway_runtime,
    write_tool_gateway_runtime_ledger,
)
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.runtime_config import agent_model_override_by_id, load_runtime_config
from sim_agent.schemas._parse import JsonMap
from sim_agent.ui.model_auth import access_token_for_provider

from .tui_parse import parse_options
from .tui_paths import display_path
from .tui_state import TuiState, append_event, replace_runtime_ledger


def handle_runtime(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    try:
        endpoint = _endpoint_from_state(state)
    except (ModelPolicyError, ProviderConfigPolicyError) as exc:
        output_stream.write(f"runtime_error={exc}\n")
        append_event(state, "runtime_blocked", str(exc))
        return state
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "runtime")))
    if _uses_tool_gateway(parsed.flags, parsed.remainder):
        return _handle_tool_gateway_runtime(endpoint, output_dir, state, output_stream)
    agent_models = _agent_model_assignments(endpoint)
    result = run_agents_sdk_runtime_dry_run(
        {
            "request_id": state.session_id,
            "user_goal": "Interactive ASA runtime session",
            "host": parsed.options.get("host", "local"),
            "estimated_runtime_s": _runtime_seconds(parsed.options.get("estimated_runtime_s")),
            "graphdb": {"mode": parsed.options.get("graphdb_mode", "dry_run")},
        },
        endpoint,
        agent_model_assignments=agent_models,
        run_sdk_smoke="smoke" in parsed.flags,
        output_dir=output_dir,
    )
    ledger = write_agents_sdk_runtime_ledger(output_dir, result)
    next_state = replace_runtime_ledger(state, ledger)
    append_event(next_state, "runtime_dry_run", result.run_id)
    output_stream.write("runtime_dry_run=true\n")
    output_stream.write(f"runtime_run_id={result.run_id}\n")
    output_stream.write(f"sdk_available={str(result.sdk_available).lower()}\n")
    output_stream.write(f"sdk_run_completed={str(result.sdk_run_completed).lower()}\n")
    for role_id in result.handoff_sequence:
        output_stream.write(f"handoff={role_id}\n")
    for role_id in result.handoff_sequence:
        model = result.agent_model_assignments[role_id]
        output_stream.write(
            f"agent_model={role_id}:{model['provider']}/{model['model']} "
            f"reasoning_effort={model['reasoning_effort']} source={model['source']}\n"
        )
    output_stream.write(f"runtime_ledger_path={display_path(ledger)}\n")
    return next_state


def _handle_tool_gateway_runtime(
    endpoint: ModelProviderConfig,
    output_dir: Path,
    state: TuiState,
    output_stream: TextIO,
) -> TuiState:
    result = run_agents_sdk_tool_gateway_runtime(
        {
            "request_id": state.session_id,
            "user_goal": "Interactive ASA runtime tool gateway session",
        },
        endpoint,
        output_dir,
        api_key=access_token_for_provider(endpoint.provider),
    )
    ledger = write_tool_gateway_runtime_ledger(output_dir, result)
    next_state = replace_runtime_ledger(state, ledger)
    append_event(next_state, "runtime_tool_gateway", result.run_id)
    output_stream.write("runtime_tool_gateway=true\n")
    output_stream.write(f"runtime_run_id={result.run_id}\n")
    output_stream.write(f"runtime_status={result.status}\n")
    output_stream.write(f"gateway_policy_id={result.gateway_policy_id}\n")
    output_stream.write(f"gateway_mode={result.gateway_mode}\n")
    output_stream.write(f"gateway_request_id={result.gateway_request_id or ''}\n")
    for tool_result in result.tool_results:
        output_stream.write(f"tool_result={tool_result.tool_name}:{tool_result.status}\n")
    for blocker in result.blockers:
        output_stream.write(f"runtime_blocker={blocker}\n")
    output_stream.write(f"runtime_ledger_path={display_path(ledger)}\n")
    return next_state


def _endpoint_from_state(state: TuiState) -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": state.model.provider,
            "model": state.model.name,
            "reasoning_effort": state.model.reasoning_effort,
            "base_url": state.model.base_url,
            "auth_mode": state.model.auth_mode,
            "api_key_env": state.model.api_key_env,
        }
    )


def _agent_model_assignments(endpoint: ModelProviderConfig) -> JsonMap:
    config = load_runtime_config()
    overrides = agent_model_override_by_id(config)
    assignments: JsonMap = {}
    for role in AGENT_ROLES:
        override = overrides.get(role.role_id)
        if override is None:
            assignments[role.role_id] = {
                "provider": endpoint.provider,
                "model": endpoint.model,
                "reasoning_effort": endpoint.reasoning_effort,
                "base_url": endpoint.base_url,
                "auth_mode": endpoint.auth_mode,
                "api_key_env": endpoint.api_key_env,
                "source": "default",
            }
            continue
        assignments[role.role_id] = {
            "provider": override.provider,
            "model": override.model,
            "reasoning_effort": override.reasoning_effort,
            "base_url": override.base_url,
            "auth_mode": override.auth_mode,
            "api_key_env": override.api_key_env,
            "source": "override",
        }
    return assignments


def _runtime_seconds(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _uses_tool_gateway(flags: tuple[str, ...], remainder: tuple[str, ...]) -> bool:
    return "tool_gateway" in flags or "tools" in remainder or "tool-gateway" in remainder
