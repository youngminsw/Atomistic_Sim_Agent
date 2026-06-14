from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TextIO

from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_gateway

from .tui_parse import parse_options
from .tui_select import MenuOption, choose_option, prompt_secret, prompt_visible
from .tui_state import TuiState, append_event


LoginMode = Literal["oauth", "api_key"]


class LoginSelector(Protocol):
    def choose_mode(self) -> LoginMode | None:
        raise NotImplementedError

    def prompt_provider(self, default: str) -> str:
        raise NotImplementedError

    def prompt_token(self, mode: LoginMode) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TerminalLoginSelector:
    input_stream: TextIO
    output_stream: TextIO

    def choose_mode(self) -> LoginMode | None:
        selected = choose_option(
            "Login Method",
            (
                MenuOption("oauth", "OAuth gateway", "browser/OAuth backed model gateway"),
                MenuOption("api_key", "API key", "direct provider token or key"),
                MenuOption("cancel", "Cancel", "return to ASA shell"),
            ),
            self.input_stream,
            self.output_stream,
        )
        match selected:
            case "oauth":
                return "oauth"
            case "api_key":
                return "api_key"
            case "cancel" | None:
                return None
            case unreachable:
                raise AssertionError(f"unexpected login mode: {unreachable}")

    def prompt_provider(self, default: str) -> str:
        return prompt_visible("Provider", default, self.input_stream, self.output_stream)

    def prompt_token(self, mode: LoginMode) -> str:
        match mode:
            case "oauth":
                return prompt_secret("OAuth access token", self.input_stream, self.output_stream)
            case "api_key":
                return prompt_secret("API key", self.input_stream, self.output_stream)


def handle_login(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    selector: LoginSelector | None = None,
) -> TuiState:
    if not args:
        if selector is not None:
            _login_interactively(selector, state, output_stream)
            return state
        write_login_options(output_stream)
        return state
    if args[0] == "help":
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


def _login_interactively(selector: LoginSelector, state: TuiState, output_stream: TextIO) -> None:
    mode = selector.choose_mode()
    if mode is None:
        output_stream.write("login_cancelled=true\n")
        return
    provider = selector.prompt_provider(state.model.provider)
    token = selector.prompt_token(mode)
    if not token:
        output_stream.write(f"login_error={mode}_token_required\n")
        return
    token_flag = "--access-token" if mode == "oauth" else "--api-key"
    _login_with_token(mode, ("--provider", provider, token_flag, token), state, output_stream)


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
