from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TextIO

from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import as_mapping, as_sequence
from sim_agent.ui.model_auth import (
    CREDENTIAL_STORE_ENV,
    login_model_provider,
    model_auth_status_payload,
)
from sim_agent.ui.model_connection import model_connection_status

from .tui_model_agents import assign_agent_model, clear_agent_model, write_agent_models
from .tui_model_catalog import choose_model, use_model, write_model_catalog
from .tui_model_endpoint import reasoning_effort_from_options
from .tui_model_profiles import (
    choose_model_profile,
    save_model_profile,
    write_active_profile_status,
    write_model_profiles,
)
from .tui_parse import parse_options
from .tui_state import ModelSettings, TuiState, append_event, replace_model
from .tui_thinking import choose_thinking_level, is_thinking_level, thinking_levels_text


def handle_model(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    input_stream: TextIO | None = None,
) -> TuiState:
    if not args:
        write_model_status(state, output_stream)
        return state
    subcommand = args[0]
    subargs = args[1:]
    match subcommand:  # noqa: MATCH_OK - user command strings are open-ended.
        case "status":
            write_model_status(state, output_stream)
            return state
        case "list":
            write_model_catalog(subargs, output_stream)
            return state
        case "use":
            return use_model(subargs, state, output_stream)
        case "set":
            return _set_model(subargs, state, output_stream, input_stream)
        case "profiles":
            write_model_profiles(output_stream)
            return state
        case "profile":
            if not subargs and input_stream is not None and input_stream.isatty():
                return choose_model_profile(state, input_stream, output_stream)
            return save_model_profile(subargs, state, output_stream)
        case "thinking" | "think":
            return _set_thinking_level(subargs, state, output_stream, input_stream)
        case "agents":
            write_agent_models(output_stream)
            return state
        case "assign":
            return assign_agent_model(subargs, state, output_stream, input_stream)
        case "clear":
            return clear_agent_model(subargs, state, output_stream)
        case "login":
            _login_model(subargs, state, output_stream)
            return state
        case _:
            output_stream.write("model_error=unknown_model_command\n")
            return state


def write_model_status(state: TuiState, output_stream: TextIO) -> None:
    runtime_config = load_runtime_config()
    connection = model_connection_status(
        state.model.provider,
        state.model.name,
        state.model.auth_mode,
        state.model.api_key_env,
    )
    output_stream.write("Model Status\n")
    output_stream.write("model_status=true\n")
    write_active_profile_status(runtime_config, output_stream)
    output_stream.write(
        f"Active model: {state.model.provider}/{state.model.name} "
        f"(thinking {state.model.reasoning_effort}, auth {state.model.auth_mode})\n"
    )
    output_stream.write(f"Connection: {connection.connection_label} - {connection.friendly_message}\n")
    output_stream.write(
        f"provider={state.model.provider} model={state.model.name} "
        f"reasoning_effort={state.model.reasoning_effort} "
        f"base_url={state.model.base_url} auth_mode={state.model.auth_mode}\n"
    )
    output_stream.write(f"model_connected={connection.connected}\n")
    output_stream.write(f"connection_label={connection.connection_label}\n")
    output_stream.write(f"model_notice={connection.friendly_message}\n")
    output_stream.write(f"model_action={connection.action_hint}\n")
    write_agent_models(output_stream)
    payload = model_auth_status_payload()
    providers = as_sequence(payload["providers"], "providers")
    if not providers:
        output_stream.write("provider=none logged_in=False\n")
        return
    for provider in providers:
        item = as_mapping(provider, "provider_status")
        output_stream.write(
            f"provider={item['provider']} logged_in={item['logged_in']} "
            f"auth_mode={item.get('auth_mode', 'oauth')} expires={item['expires']}\n"
        )


def _set_model(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    input_stream: TextIO | None,
) -> TuiState:
    if not args:
        return choose_model(state, output_stream, input_stream)
    parsed = parse_options(args)
    options = parsed.options
    model = ModelSettings(
        provider=options.get("provider", state.model.provider),
        name=options.get("model", state.model.name),
        reasoning_effort=reasoning_effort_from_options(options, state.model.reasoning_effort),
        base_url=options.get("base_url", state.model.base_url),
        auth_mode=options.get("auth_mode", state.model.auth_mode),
        api_key_env=options.get("api_key_env", state.model.api_key_env),
    )
    next_state = replace_model(state, model)
    append_event(next_state, "model_set", f"{model.provider}/{model.name}/{model.auth_mode}")
    output_stream.write("model_set_ok=true\n")
    output_stream.write(
        f"provider={model.provider} model={model.name} "
        f"reasoning_effort={model.reasoning_effort} auth_mode={model.auth_mode}\n"
    )
    return next_state


def _set_thinking_level(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    input_stream: TextIO | None,
) -> TuiState:
    if not args and input_stream is not None and input_stream.isatty():
        selected = choose_thinking_level("Model Thinking Level", state.model.reasoning_effort, input_stream, output_stream)
        if selected is None:
            output_stream.write("model_thinking_selection_cancelled=true\n")
            return state
        return _save_thinking_level(selected, state, output_stream)
    parsed = parse_options(args)
    level = reasoning_effort_from_options(
        parsed.options,
        parsed.remainder[0] if parsed.remainder else state.model.reasoning_effort,
    )
    return _save_thinking_level(level, state, output_stream)


def _save_thinking_level(level: str, state: TuiState, output_stream: TextIO) -> TuiState:
    if not is_thinking_level(level):
        output_stream.write(f"model_error=unknown_thinking_level; allowed={thinking_levels_text()}\n")
        return state
    model = ModelSettings(
        provider=state.model.provider,
        name=state.model.name,
        reasoning_effort=level,
        base_url=state.model.base_url,
        auth_mode=state.model.auth_mode,
        api_key_env=state.model.api_key_env,
    )
    next_state = replace_model(state, model)
    append_event(next_state, "model_thinking_level_set", level)
    output_stream.write(f"Thinking level saved: {level}\n")
    output_stream.write("model_thinking_level_saved=true\n")
    output_stream.write(f"thinking_level={level} reasoning_effort={level}\n")
    return next_state


def _login_model(args: Sequence[str], state: TuiState, output_stream: TextIO) -> None:
    parsed = parse_options(args)
    options = parsed.options
    credential_store = options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store
    access_token = options.get("access_token")
    if access_token is None:
        output_stream.write("model_login_error=access_token_required\n")
        return
    payload = login_model_provider(
        {
            "provider": options.get("provider", state.model.provider),
            "access_token": access_token,
            "refresh_token": options.get("refresh_token"),
            "expires_in_s": 3600,
        }
    )
    append_event(state, "model_login", f"provider={payload['provider']}")
    output_stream.write("model_login_ok=true\n")
    output_stream.write(f"provider={payload['provider']}\n")
    output_stream.write(f"provider_credential_store={payload['provider_credential_store']}\n")
