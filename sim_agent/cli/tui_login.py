from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TextIO

from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_gateway

from .tui_parse import parse_options
from .tui_state import TuiState, append_event


def handle_login(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    if not args or args[0] == "help":
        write_login_options(output_stream)
        return state
    mode = args[0]
    subargs = args[1:]
    match mode:
        case "oauth":
            _login_with_token("oauth", subargs, state, output_stream)
        case "api-key" | "apikey":
            _login_with_token("api_key", subargs, state, output_stream)
        case _:
            output_stream.write("login_error=unknown_login_mode\n")
            write_login_options(output_stream)
    return state


def write_login_options(output_stream: TextIO) -> None:
    output_stream.write("Login Options\n")
    output_stream.write("login_selector=true\n")
    output_stream.write("/login oauth --provider <id> --access-token <token> [--refresh-token <token>]\n")
    output_stream.write("/login api-key --provider <id> --api-key <token>\n")
    output_stream.write("/login api-key --provider <id> --api-key-env <ENV_WITH_TOKEN>\n")
    output_stream.write("tip=/model set --provider <id> --model <model> --auth-mode oauth|api_key|gateway\n")


def _login_with_token(mode: str, args: Sequence[str], state: TuiState, output_stream: TextIO) -> None:
    parsed = parse_options(args)
    credential_store = parsed.options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store
    provider = parsed.options.get("provider", state.model.provider)
    token = _token(parsed.options, mode)
    if token is None:
        output_stream.write(f"login_error={mode}_token_required\n")
        write_login_options(output_stream)
        return
    payload = login_model_gateway(
        {
            "provider": provider,
            "access_token": token,
            "refresh_token": parsed.options.get("refresh_token", token),
            "auth_mode": mode,
            "expires_in_s": 3600,
        }
    )
    append_event(state, "model_login", f"provider={payload['provider']} auth_mode={mode}")
    output_stream.write("login_ok=true\n")
    output_stream.write(f"provider={payload['provider']} auth_mode={mode}\n")
    output_stream.write(f"credential_store={payload['credential_store']}\n")


def _token(options: dict[str, str], mode: str) -> str | None:
    direct = options.get("access_token") or options.get("api_key")
    if direct:
        return direct
    env_name = options.get("api_key_env")
    if mode == "api_key" and env_name:
        value = os.environ.get(env_name)
        if value:
            return value
    return None
