from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TextIO

from sim_agent.agents_sdk_runtime.roles import AGENT_ROLES
from sim_agent.llm_endpoints import (
    ModelPolicyError,
    ModelUseCase,
    ProviderConfigPolicyError,
    find_model_catalog_entry,
    list_model_catalog,
    model_catalog_references,
)
from sim_agent.runtime_config import (
    AgentModelRuntimeConfig,
    ModelEndpointRuntimeConfig,
    RuntimeConfig,
    agent_model_override_by_id,
    load_runtime_config,
    mark_active_profile_customized,
    remove_agent_model_override,
    save_runtime_config,
    upsert_agent_model_override,
)

from .tui_model_endpoint import endpoint_from_command, validated_endpoint
from .tui_parse import parse_options
from .tui_select import MenuOption, choose_option
from .tui_state import TuiState, append_event
from .tui_thinking import choose_thinking_level


def write_agent_models(output_stream: TextIO) -> None:
    config = load_runtime_config()
    overrides = agent_model_override_by_id(config)
    output_stream.write("agent_models=true\n")
    for role in AGENT_ROLES:
        override = overrides.get(role.role_id)
        endpoint = override or config.model_endpoint
        write_agent_model_row(role.role_id, endpoint, override is not None, output_stream)


def assign_agent_model(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    input_stream: TextIO | None = None,
) -> TuiState:
    if not args and input_stream is not None and input_stream.isatty():
        return _assign_agent_model_interactive(state, input_stream, output_stream)
    parsed = parse_options(args)
    agent_id = parsed.options.get("agent")
    if agent_id is None or agent_id not in agent_ids():
        output_stream.write(f"model_error=unknown_agent; allowed={','.join(agent_ids())}\n")
        return state
    config = load_runtime_config()
    endpoint = endpoint_from_command(parsed.remainder, parsed.options, current_agent_endpoint(config, agent_id))
    if not endpoint.model:
        output_stream.write("model_error=model_reference_required\n")
        output_stream.write(f"model_catalog_refs={model_catalog_references()}\n")
        return state
    try:
        normalized = validated_endpoint(endpoint, ModelUseCase.LOW_RISK_SUMMARIZATION)
    except (ModelPolicyError, ProviderConfigPolicyError) as exc:
        output_stream.write(f"agent_model_error={exc}\n")
        append_event(state, "agent_model_blocked", f"{agent_id}:{exc}")
        return state

    override = AgentModelRuntimeConfig(
        agent_id=agent_id,
        provider=normalized.provider,
        model=normalized.model,
        reasoning_effort=normalized.reasoning_effort,
        base_url=normalized.base_url,
        auth_mode=normalized.auth_mode,
        api_key_env=normalized.api_key_env,
    )
    overrides = upsert_agent_model_override(config.agent_model_overrides, override)
    path = save_runtime_config(mark_active_profile_customized(replace(config, agent_model_overrides=overrides)))
    append_event(state, "agent_model_saved", f"{agent_id}:{override.provider}/{override.model}")
    output_stream.write(f"agent_model_saved={agent_id}\n")
    write_agent_model_row(agent_id, override, True, output_stream)
    output_stream.write(f"runtime_config_path={path}\n")
    return state


def _assign_agent_model_interactive(state: TuiState, input_stream: TextIO, output_stream: TextIO) -> TuiState:
    agent_id = choose_option(
        "Agent",
        tuple(MenuOption(role.role_id, role.display_name, role.boundary) for role in AGENT_ROLES),
        input_stream,
        output_stream,
    )
    if agent_id is None:
        output_stream.write("agent_model_selection_cancelled=true\n")
        return state
    model_ref = choose_option(
        "Agent Model",
        tuple(
            MenuOption(
                entry.reference,
                f"{entry.company} / {entry.source_provider}",
                f"{entry.model} · {entry.reasoning_effort} · {entry.role_hint}",
            )
            for entry in list_model_catalog()
        ),
        input_stream,
        output_stream,
    )
    if model_ref is None:
        output_stream.write("agent_model_selection_cancelled=true\n")
        return state
    entry = find_model_catalog_entry(model_ref)
    default_level = entry.reasoning_effort if entry is not None else "medium"
    thinking = choose_thinking_level("Agent Thinking Level", default_level, input_stream, output_stream)
    if thinking is None:
        output_stream.write("agent_thinking_selection_cancelled=true\n")
        return state
    return assign_agent_model(("--agent", agent_id, model_ref, "--thinking-level", thinking), state, output_stream)


def clear_agent_model(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    agent_id = parsed.options.get("agent")
    if agent_id is None or agent_id not in agent_ids():
        output_stream.write(f"model_error=unknown_agent; allowed={','.join(agent_ids())}\n")
        return state
    config = load_runtime_config()
    overrides = remove_agent_model_override(config.agent_model_overrides, agent_id)
    path = save_runtime_config(mark_active_profile_customized(replace(config, agent_model_overrides=overrides)))
    append_event(state, "agent_model_cleared", agent_id)
    output_stream.write(f"agent_model_cleared={agent_id}\n")
    output_stream.write(f"runtime_config_path={path}\n")
    return state


def write_agent_model_row(
    agent_id: str,
    endpoint: ModelEndpointRuntimeConfig | AgentModelRuntimeConfig,
    is_override: bool,
    output_stream: TextIO,
) -> None:
    output_stream.write(
        f"agent={agent_id} provider={endpoint.provider} model={endpoint.model} "
        f"reasoning_effort={endpoint.reasoning_effort} override={str(is_override).lower()} "
        f"auth_mode={endpoint.auth_mode} base_url={endpoint.base_url}\n"
    )


def current_agent_endpoint(config: RuntimeConfig, agent_id: str) -> ModelEndpointRuntimeConfig:
    override = agent_model_override_by_id(config).get(agent_id)
    if override is None:
        return config.model_endpoint
    return ModelEndpointRuntimeConfig(
        provider=override.provider,
        model=override.model,
        reasoning_effort=override.reasoning_effort,
        base_url=override.base_url,
        auth_mode=override.auth_mode,
        api_key_env=override.api_key_env,
    )


def agent_ids() -> tuple[str, ...]:
    return tuple(role.role_id for role in AGENT_ROLES)
