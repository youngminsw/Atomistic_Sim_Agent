from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TextIO

from sim_agent.llm_endpoints.model_catalog import find_model_catalog_entry
from sim_agent.llm_endpoints.model_profiles import (
    ModelProfile,
    ModelProfileAssignment,
    find_model_profile,
    list_model_profiles,
)
from sim_agent.provider_registry import OPENAI_CODEX_BASE_URL, OPENAI_CODEX_TOKEN_ENV, provider_by_id
from sim_agent.runtime_config import (
    ActiveModelProfileRuntimeConfig,
    AgentModelRuntimeConfig,
    ModelEndpointRuntimeConfig,
    RuntimeConfig,
    active_profile_status,
    load_runtime_config,
    save_runtime_config,
)

from .tui_model_endpoint import model_settings_from_endpoint, write_endpoint_config
from .tui_select import MenuOption, choose_option
from .tui_state import TuiState, append_event, replace_model


def write_model_profiles(output_stream: TextIO) -> None:
    profiles = list_model_profiles()
    if output_stream.isatty():
        _write_human_model_profiles(profiles, output_stream)
        return
    output_stream.write("model_profiles=true\n")
    for profile in profiles:
        _write_profile(profile, output_stream)


def choose_model_profile(state: TuiState, input_stream: TextIO, output_stream: TextIO) -> TuiState:
    selected = choose_option(
        "Model Profile",
        tuple(
            MenuOption(
                value=profile.name,
                label=profile.label,
                summary=f"{profile.default.reference}:{profile.default.reasoning_effort} · {profile.summary}",
            )
            for profile in list_model_profiles()
        ),
        input_stream,
        output_stream,
    )
    if selected is None:
        output_stream.write("model_profile_selection_cancelled=true\n")
        return state
    return save_model_profile((selected,), state, output_stream)


def save_model_profile(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    if not args:
        output_stream.write("model_profile_error=profile_required\n")
        output_stream.write(f"model_profiles_available={_profile_names()}\n")
        return state
    profile = find_model_profile(args[0])
    if profile is None:
        output_stream.write(f"model_profile_error=unknown_profile; allowed={_profile_names()}\n")
        return state
    runtime_config = load_runtime_config()
    next_config = _config_from_profile(profile, runtime_config)
    path = save_runtime_config(next_config)
    next_state = replace_model(state, model_settings_from_endpoint(next_config.model_endpoint))
    append_event(next_state, "model_profile_saved", profile.name)
    output_stream.write(f"model_profile_saved={profile.name}\n")
    _write_active_profile(next_config, output_stream)
    write_endpoint_config(next_config.model_endpoint, output_stream)
    for override in next_config.agent_model_overrides:
        _write_profile_agent_row(override, output_stream)
    output_stream.write(f"runtime_config_path={path}\n")
    return next_state


def _write_human_model_profiles(profiles: Sequence[ModelProfile], output_stream: TextIO) -> None:
    output_stream.write("\nModel Profiles\n")
    for profile in profiles:
        output_stream.write(
            f"  {profile.name:<14} {profile.default.reference}:{profile.default.reasoning_effort:<7} "
            f"{profile.summary}\n"
        )
        for assignment in profile.agents:
            output_stream.write(
                f"    @{assignment.agent_id:<24} {assignment.reference}:{assignment.reasoning_effort}\n"
            )
    output_stream.write("\nUse /model profile <name> or open /model set and choose a profile.\n")


def _write_profile(profile: ModelProfile, output_stream: TextIO) -> None:
    output_stream.write(f"model_profile={profile.name} label={profile.label}\n")
    output_stream.write(f"default={profile.default.reference}:{profile.default.reasoning_effort}\n")
    output_stream.write(f"profile_summary={profile.summary}\n")
    for assignment in profile.agents:
        output_stream.write(
            f"agent_model={assignment.agent_id}:{assignment.reference}:{assignment.reasoning_effort}\n"
        )


def _write_profile_agent_row(override: AgentModelRuntimeConfig, output_stream: TextIO) -> None:
    output_stream.write(
        f"agent_model={override.agent_id}:{override.provider}/{override.model}:{override.reasoning_effort}\n"
    )


def write_active_profile_status(config: RuntimeConfig, output_stream: TextIO) -> None:
    _write_active_profile(config, output_stream)


def _write_active_profile(config: RuntimeConfig, output_stream: TextIO) -> None:
    active = active_profile_status(config)
    profile_name = active.name or "none"
    output_stream.write(f"active_profile={profile_name}\n")
    output_stream.write(f"profile_customized={str(active.customized).lower()}\n")
    output_stream.write(f"model_profile={profile_name} customized={str(active.customized).lower()}\n")


def _config_from_profile(profile: ModelProfile, runtime_config: RuntimeConfig) -> RuntimeConfig:
    endpoint = _endpoint_from_assignment(profile.default)
    overrides = tuple(_override_from_assignment(assignment) for assignment in profile.agents)
    return replace(
        runtime_config,
        model_endpoint=endpoint,
        active_profile=ActiveModelProfileRuntimeConfig(name=profile.name, customized=False),
        agent_model_overrides=overrides,
    )


def _endpoint_from_assignment(assignment: ModelProfileAssignment) -> ModelEndpointRuntimeConfig:
    entry = find_model_catalog_entry(assignment.reference)
    if entry is None:
        spec = provider_by_id(assignment.provider)
        base_url = spec.default_base_url if spec is not None else OPENAI_CODEX_BASE_URL
        auth_mode = spec.default_auth_mode if spec is not None else "oauth"
        api_key_env = spec.default_api_key_env if spec is not None else OPENAI_CODEX_TOKEN_ENV
        return ModelEndpointRuntimeConfig(
            provider=assignment.provider,
            model=assignment.model,
            reasoning_effort=assignment.reasoning_effort,
            base_url=base_url,
            auth_mode=auth_mode,
            api_key_env=api_key_env,
        )
    return ModelEndpointRuntimeConfig(
        provider=entry.provider,
        model=entry.model,
        reasoning_effort=assignment.reasoning_effort,
        base_url=entry.base_url,
        auth_mode=entry.auth_mode,
        api_key_env=entry.api_key_env,
    )


def _override_from_assignment(assignment: ModelProfileAssignment) -> AgentModelRuntimeConfig:
    endpoint = _endpoint_from_assignment(assignment)
    return AgentModelRuntimeConfig(
        agent_id=assignment.agent_id,
        provider=endpoint.provider,
        model=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )


def _profile_names() -> str:
    return ",".join(profile.name for profile in list_model_profiles())
