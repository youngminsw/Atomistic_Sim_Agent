from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TextIO

from sim_agent.schemas._parse import as_mapping, as_sequence
from sim_agent.ui.model_auth import (
    CREDENTIAL_STORE_ENV,
    login_model_gateway,
    model_auth_status_payload,
)
from sim_agent.ui.model_connection import model_connection_status

from .tui_parse import parse_options
from .tui_state import ModelSettings, TuiState, append_event, replace_model


def handle_model(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    if not args:
        write_model_status(state, output_stream)
        return state
    subcommand = args[0]
    subargs = args[1:]
    match subcommand:
        case "status":
            write_model_status(state, output_stream)
            return state
        case "set":
            return _set_model(subargs, state, output_stream)
        case "login":
            _login_model(subargs, state, output_stream)
            return state
        case _:
            output_stream.write("model_error=unknown_model_command\n")
            return state


def write_model_status(state: TuiState, output_stream: TextIO) -> None:
    connection = model_connection_status(
        state.model.provider,
        state.model.name,
        state.model.auth_mode,
        state.model.api_key_env,
    )
    output_stream.write("model_status=true\n")
    output_stream.write(
        f"provider={state.model.provider} model={state.model.name} "
        f"base_url={state.model.base_url} auth_mode={state.model.auth_mode}\n"
    )
    output_stream.write(f"model_connected={connection.connected}\n")
    output_stream.write(f"connection_label={connection.connection_label}\n")
    output_stream.write(f"model_notice={connection.friendly_message}\n")
    output_stream.write(f"model_action={connection.action_hint}\n")
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


def _set_model(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    options = parsed.options
    model = ModelSettings(
        provider=options.get("provider", state.model.provider),
        name=options.get("model", state.model.name),
        base_url=options.get("base_url", state.model.base_url),
        auth_mode=options.get("auth_mode", state.model.auth_mode),
        api_key_env=options.get("api_key_env", state.model.api_key_env),
    )
    next_state = replace_model(state, model)
    append_event(next_state, "model_set", f"{model.provider}/{model.name}/{model.auth_mode}")
    output_stream.write("model_set_ok=true\n")
    output_stream.write(f"provider={model.provider} model={model.name} auth_mode={model.auth_mode}\n")
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
    payload = login_model_gateway(
        {
            "provider": options.get("provider", "oauth_gateway"),
            "access_token": access_token,
            "refresh_token": options.get("refresh_token"),
            "expires_in_s": 3600,
        }
    )
    append_event(state, "model_login", f"provider={payload['provider']}")
    output_stream.write("model_login_ok=true\n")
    output_stream.write(f"provider={payload['provider']}\n")
    output_stream.write(f"credential_store={payload['credential_store']}\n")
