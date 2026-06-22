from __future__ import annotations

import os
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TextIO

from sim_agent.provider_registry import login_profile_by_id
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_gateway

from .tui_browser_oauth import start_browser_oauth
from .tui_login_profiles import LOGIN_PROFILES, LoginProfile, LoginTarget, choose_login_target, login_companies
from .tui_parse import parse_options
from .tui_redaction import machine_output, write_login_success
from .tui_select import prompt_secret
from .tui_state import TuiState, append_event


class LoginSelector(Protocol):
    def choose_target(self, default_provider: str) -> LoginTarget | None:
        raise NotImplementedError

    def prompt_token(self, target: LoginTarget) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TerminalLoginSelector:
    input_stream: TextIO
    output_stream: TextIO

    def choose_target(self, default_provider: str) -> LoginTarget | None:
        return choose_login_target(default_provider, self.input_stream, self.output_stream)

    def prompt_token(self, target: LoginTarget) -> str:
        match target.token_mode:
            case "oauth":
                return ""
            case "api_key":
                _write_api_key_dashboard(target, self.output_stream)
                return prompt_secret(f"{target.label} API key", self.input_stream, self.output_stream)


def handle_login(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    selector: LoginSelector | None = None,
) -> TuiState:
    if not args:
        if selector is not None:
            return _login_interactively(selector, state, output_stream)
        write_login_options(output_stream)
        return state
    if args[0] == "help":
        write_login_options(output_stream)
        return state
    mode = args[0]
    subargs = args[1:]
    match mode:
        case "oauth":
            parsed = parse_options(subargs)
            if _token(parsed.options, "oauth") is None:
                target = _oauth_target(parsed.options, state)
                start_browser_oauth(target, subargs, state, output_stream)
            else:
                _login_with_token("oauth", subargs, state, output_stream)
        case "api-key" | "apikey":
            parsed = parse_options(subargs)
            target = _api_key_target(parsed.options, state)
            if _token(parsed.options, "api_key") is None:
                _write_api_key_dashboard(target, output_stream, parsed.flags)
            _login_with_token("api_key", subargs, state, output_stream)
        case _:
            output_stream.write("login_error=unknown_login_mode\n")
            write_login_options(output_stream)
    return state


def _login_interactively(selector: LoginSelector, state: TuiState, output_stream: TextIO) -> TuiState:
    target = selector.choose_target(state.model.provider)
    if target is None:
        output_stream.write("login_cancelled=true\n")
        return state
    profile_spec = login_profile_by_id(target.profile)
    if profile_spec is not None and profile_spec.flow_kind == "local":
        output_stream.write("login_not_required=true\n")
        output_stream.write(f"provider={target.provider} auth_mode=none\n")
        append_event(state, "model_login_skipped", f"provider={target.provider} local")
        return state
    if target.token_mode == "oauth":
        start_browser_oauth(target, (), state, output_stream)
        return state
    token = selector.prompt_token(target)
    if not token:
        output_stream.write(f"login_error={target.token_mode}_token_required\n")
        return state
    token_flag = "--access-token" if target.token_mode == "oauth" else "--api-key"
    _login_with_token(
        target.token_mode,
        ("--provider", target.provider, "--profile", target.profile, token_flag, token),
        state,
        output_stream,
    )
    return state


def write_login_options(output_stream: TextIO) -> None:
    output_stream.write("Login Options\n")
    output_stream.write("login_selector=true\n")
    for company in login_companies():
        output_stream.write(f"login_company={company}\n")
    for profile in LOGIN_PROFILES:
        profile_spec = login_profile_by_id(profile.value)
        flow_kind = profile_spec.flow_kind if profile_spec is not None else profile.token_mode
        output_stream.write(
            f"login_profile={profile.value} login_company={profile.company} "
            f"login_provider={profile.provider}\n"
        )
        output_stream.write(
            f"login_auth label={profile.label} "
            f"auth_mode={profile.token_mode} flow={flow_kind}\n"
        )
        output_stream.write(f"login_summary={profile.summary}\n")
        if profile_spec is not None and profile_spec.dashboard_url:
            output_stream.write(f"login_dashboard={profile.value} url={profile_spec.dashboard_url}\n")
    output_stream.write("browser_oauth_first=true\n")
    output_stream.write("/login oauth --provider <id> [--auth-url <url>] [--no-open]\n")
    output_stream.write("/login oauth --provider <id> --access-token <token> [--refresh-token <token>]  # fallback\n")
    output_stream.write("/login api-key --provider <id> --api-key <token>\n")
    output_stream.write("/login api-key --provider <id> --api-key-env <ENV_WITH_TOKEN>\n")
    output_stream.write("tip=/model use <provider/model> or /model assign --agent <id> <provider/model>\n")


def _login_with_token(mode: str, args: Sequence[str], state: TuiState, output_stream: TextIO) -> None:
    parsed = parse_options(args)
    credential_store = parsed.options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store
    provider = parsed.options.get("provider", state.model.provider)
    profile = parsed.options.get("profile")
    token = _token(parsed.options, mode)
    if token is None:
        output_stream.write(f"login_error={mode}_token_required\n")
        write_login_options(output_stream)
        return
    payload = login_model_gateway(
        {
            "provider": provider,
            "login_profile": profile,
            "access_token": token,
            "refresh_token": parsed.options.get("refresh_token", token),
            "auth_mode": mode,
            "expires_in_s": 3600,
        }
    )
    append_event(state, "model_login", f"provider={payload['provider']} auth_mode={mode}")
    if machine_output(output_stream):
        output_stream.write("login_ok=true\n")
        output_stream.write(f"provider={payload['provider']} auth_mode={mode}\n")
        if profile:
            output_stream.write(f"login_profile={profile}\n")
        output_stream.write(f"credential_store={payload['credential_store']}\n")
        return
    write_login_success(output_stream, provider=payload["provider"], label=_login_success_label(profile, provider))


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


def _login_success_label(profile: str | None, provider: str) -> str:
    if profile is None:
        return provider
    profile_spec = login_profile_by_id(profile)
    if profile_spec is None:
        return provider
    return profile_spec.label


def _oauth_target(options: dict[str, str], state: TuiState) -> LoginTarget:
    profile_id = options.get("profile")
    provider = options.get("provider")
    profile = _oauth_profile(profile_id, provider)
    if profile is not None:
        return LoginTarget(profile.value, profile.provider, "oauth", profile.label, profile.company)
    selected_provider = provider or state.model.provider
    selected_profile = profile_id or selected_provider
    return LoginTarget(selected_profile, selected_provider, "oauth", selected_provider, "Custom")


def _api_key_target(options: dict[str, str], state: TuiState) -> LoginTarget:
    profile_id = options.get("profile")
    provider = options.get("provider") or state.model.provider
    for profile in LOGIN_PROFILES:
        if profile.token_mode != "api_key":
            continue
        if profile_id == profile.value or provider == profile.provider:
            return LoginTarget(profile.value, profile.provider, "api_key", profile.label, profile.company)
    return LoginTarget(profile_id or provider, provider, "api_key", provider, "Custom")


def _oauth_profile(profile_id: str | None, provider: str | None) -> LoginProfile | None:
    for profile in LOGIN_PROFILES:
        if profile.token_mode != "oauth":
            continue
        if profile_id == profile.value or provider == profile.provider:
            return profile
    return None


def _write_api_key_dashboard(
    target: LoginTarget,
    output_stream: TextIO,
    flags: Sequence[str] = (),
) -> None:
    profile = login_profile_by_id(target.profile)
    if profile is None or not profile.dashboard_url:
        return
    opened = _open_browser(profile.dashboard_url, flags)
    output_stream.write(f"api_key_dashboard_url={profile.dashboard_url}\n")
    output_stream.write(f"api_key_dashboard_opened={opened}\n")


def _open_browser(url: str, flags: Sequence[str]) -> bool:
    disabled = os.environ.get("ASA_BROWSER_OAUTH_OPEN", "").lower() in {"0", "false", "no", "off"}
    if disabled or "no_open" in flags:
        return False
    try:
        return bool(webbrowser.open(url, new=1, autoraise=True))
    except webbrowser.Error:
        return False
