from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO
from urllib.parse import quote, urlparse

from sim_agent.provider_registry import login_profile_by_id, login_profile_by_provider
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

from .tui_browser_launch import open_url_in_browser, write_oauth_browser_block
from .tui_kimi_oauth import KimiOAuthOutcome, start_kimi_device_oauth
from .tui_login_profiles import LoginTarget
from .tui_openai_codex_oauth import OAuthFlowOutcome, start_openai_codex_oauth
from .tui_parse import parse_options
from .tui_redaction import machine_output
from .tui_state import TuiState, append_event


GLOBAL_AUTH_URL_ENV = "ASA_OAUTH_GATEWAY_AUTH_URL"


@dataclass(frozen=True, slots=True)
class BrowserOAuthResult:
    provider: str
    profile: str
    started: bool
    opened: bool
    url: str | None = None
    blocker: str | None = None


def start_browser_oauth(
    target: LoginTarget,
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
) -> BrowserOAuthResult:
    parsed = parse_options(args)
    credential_store = parsed.options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store
    auth_url = _configured_auth_url(target, parsed.options)
    if auth_url is not None:
        return _start_configured_provider_oauth(target, auth_url, parsed.flags, state, output_stream, credential_store)
    if _is_openai_codex_target(target):
        return _from_provider_outcome(start_openai_codex_oauth(target, tuple(args), state, output_stream))
    if _is_kimi_code_target(target):
        return _from_kimi_outcome(start_kimi_device_oauth(target, tuple(args), state, output_stream))
    profile = login_profile_by_id(target.profile) or login_profile_by_provider(target.provider)
    if profile is not None and profile.flow_kind == "manual_oauth" and profile.dashboard_url:
        return _start_manual_provider_oauth(target, profile.dashboard_url, parsed.flags, state, output_stream)
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=false\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write("browser_oauth_blocker=missing_provider_oauth_flow\n")
        output_stream.write("browser_oauth_next=provider exists in registry; add native OAuth controller or pass --auth-url for a custom gateway\n")
    else:
        output_stream.write("OAuth login is not wired for this provider yet.\n")
        output_stream.write("Next: pass --auth-url for a custom gateway or choose an API-key profile.\n")
    append_event(state, "browser_oauth_blocked", f"provider={target.provider} missing_provider_oauth_flow")
    return BrowserOAuthResult(
        provider=target.provider,
        profile=target.profile,
        started=False,
        opened=False,
        blocker="missing_provider_oauth_flow",
    )


def _from_provider_outcome(outcome: OAuthFlowOutcome) -> BrowserOAuthResult:
    return BrowserOAuthResult(
        provider=outcome.provider,
        profile=outcome.profile,
        started=outcome.started,
        opened=outcome.opened,
        url=outcome.url,
        blocker=outcome.blocker,
    )


def _from_kimi_outcome(outcome: KimiOAuthOutcome) -> BrowserOAuthResult:
    return BrowserOAuthResult(
        provider=outcome.provider,
        profile=outcome.profile,
        started=outcome.started,
        opened=outcome.opened,
        url=outcome.url,
        blocker=outcome.blocker,
    )


def _start_configured_provider_oauth(
    target: LoginTarget,
    auth_url: str,
    flags: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    credential_store: str | None,
) -> BrowserOAuthResult:
    if not _is_http_url(auth_url):
        if machine_output(output_stream):
            output_stream.write("browser_oauth_started=false\n")
            output_stream.write(f"browser_oauth_provider={target.provider}\n")
            output_stream.write(f"browser_oauth_profile={target.profile}\n")
            output_stream.write("browser_oauth_blocker=invalid_oauth_auth_url\n")
        else:
            output_stream.write("OAuth login could not start: invalid auth URL.\n")
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} invalid_oauth_auth_url")
        return BrowserOAuthResult(target.provider, target.profile, False, False, blocker="invalid_oauth_auth_url")
    opened = _open_browser(auth_url, flags)
    write_oauth_browser_block(
        output_stream,
        url=auth_url,
        opened=opened,
        instructions="Complete browser login; provider callback must write the provider credential store.",
    )
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=true\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_url={auth_url}\n")
        output_stream.write(f"browser_opened={opened}\n")
        output_stream.write("browser_oauth_credential_saved=false\n")
        output_stream.write("browser_oauth_next=complete browser login; provider callback must write provider credential store\n")
        if credential_store:
            output_stream.write(f"provider_credential_store={credential_store}\n")
    else:
        output_stream.write("Waiting for browser login to finish.\n")
        output_stream.write("Next: return to ASA after the provider confirms the login.\n")
    append_event(state, "browser_oauth_started", f"provider={target.provider} opened={opened}")
    return BrowserOAuthResult(target.provider, target.profile, True, opened, auth_url)


def _start_manual_provider_oauth(
    target: LoginTarget,
    provider_url: str,
    flags: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
) -> BrowserOAuthResult:
    opened = _open_browser(provider_url, flags)
    write_oauth_browser_block(
        output_stream,
        url=provider_url,
        opened=opened,
        instructions="Native token exchange is not wired for this provider yet; use API key flow where available.",
    )
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=true\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_url={provider_url}\n")
        output_stream.write(f"browser_opened={opened}\n")
        output_stream.write("browser_oauth_credential_saved=false\n")
        output_stream.write("browser_oauth_next=native provider token exchange is not wired yet; use API key flow where available\n")
    else:
        output_stream.write("Native token exchange is not wired for this provider yet.\n")
        output_stream.write("Next: use API key login where available.\n")
    append_event(state, "browser_oauth_started", f"provider={target.provider} manual_provider_flow")
    return BrowserOAuthResult(target.provider, target.profile, True, opened, provider_url, "manual_provider_flow")


def _is_openai_codex_target(target: LoginTarget) -> bool:
    return target.provider == "openai-codex" and target.profile in {"chatgpt_codex", "chatgpt_codex_device"}


def _is_kimi_code_target(target: LoginTarget) -> bool:
    return target.provider == "kimi-code" and target.profile == "kimi_code"


def _configured_auth_url(target: LoginTarget, options: dict[str, str]) -> str | None:
    direct = options.get("auth_url")
    if direct:
        return _expand_url(direct, target)
    for env_name in _auth_url_env_names(target):
        value = os.environ.get(env_name)
        if value:
            return _expand_url(value, target)
    return None


def _auth_url_env_names(target: LoginTarget) -> tuple[str, ...]:
    profile_suffix = _env_suffix(target.profile)
    provider_suffix = _env_suffix(target.provider)
    return (
        f"ASA_OAUTH_AUTH_URL_{profile_suffix}",
        f"ASA_OAUTH_AUTH_URL_{provider_suffix}",
        GLOBAL_AUTH_URL_ENV,
    )


def _env_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _expand_url(value: str, target: LoginTarget) -> str:
    return value.replace("{provider}", quote(target.provider, safe="")).replace("{profile}", quote(target.profile, safe=""))


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _open_browser(url: str, flags: Sequence[str]) -> bool:
    return open_url_in_browser(url, flags)
