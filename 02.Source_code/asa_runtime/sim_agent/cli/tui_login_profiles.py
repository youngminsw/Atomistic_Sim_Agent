from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TextIO

from sim_agent.provider_registry import login_companies as registry_login_companies
from sim_agent.provider_registry import login_profile_specs

from .tui_select import MenuOption, choose_option


LoginMode = Literal["oauth", "api_key"]


@dataclass(frozen=True, slots=True)
class LoginProfile:
    value: str
    company: str
    label: str
    summary: str
    provider: str
    token_mode: LoginMode


@dataclass(frozen=True, slots=True)
class LoginTarget:
    profile: str
    provider: str
    token_mode: LoginMode
    label: str
    company: str


LOGIN_PROFILES: tuple[LoginProfile, ...] = tuple(
    LoginProfile(
        spec.profile_id,
        spec.company,
        spec.label,
        spec.summary,
        spec.provider_id,
        spec.token_mode,
    )
    for spec in login_profile_specs()
)


def choose_login_target(
    default_provider: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> LoginTarget | None:
    company = _choose_company(default_provider, input_stream, output_stream)
    if company is None:
        return None
    selected = choose_option(
        "Login Provider",
        tuple(
            MenuOption(profile.value, profile.label, profile.summary)
            for profile in LOGIN_PROFILES
            if profile.company == company
        )
        + (MenuOption("cancel", "Cancel", "return to ASA shell"),),
        input_stream,
        output_stream,
    )
    if selected in {None, "cancel"}:
        return None
    profile = _profile_by_value(selected)
    return LoginTarget(
        profile=profile.value,
        provider=profile.provider,
        token_mode=profile.token_mode,
        label=profile.label,
        company=profile.company,
    )


def login_companies() -> tuple[str, ...]:
    return registry_login_companies()


def _choose_company(default_provider: str, input_stream: TextIO, output_stream: TextIO) -> str | None:
    selected = choose_option(
        "Login Company",
        tuple(MenuOption(company, company, _company_summary(company, default_provider)) for company in login_companies())
        + (MenuOption("cancel", "Cancel", "return to ASA shell"),),
        input_stream,
        output_stream,
    )
    if selected in {None, "cancel"}:
        return None
    return selected


def _company_summary(company: str, default_provider: str) -> str:
    profiles = tuple(profile for profile in LOGIN_PROFILES if profile.company == company)
    if any(profile.provider == default_provider for profile in profiles):
        return "current login provider group"
    return ", ".join(profile.label for profile in profiles[:3])


def _profile_by_value(value: str) -> LoginProfile:
    for profile in LOGIN_PROFILES:
        if profile.value == value:
            return profile
    return LoginProfile("current", "Custom", value, "current provider", value, "oauth")
