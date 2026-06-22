from __future__ import annotations

import ipaddress
import os
import secrets
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Final, Protocol
from urllib.parse import urlparse

from sim_agent.schemas._parse import JsonMap


CONTROLLER_TOKEN_ENV: Final = "ASA_CONTROLLER_TOKEN"


class HeaderMapping(Protocol):
    def get(self, name: str, default: str = "") -> str | None: ...


@dataclass(frozen=True, slots=True)
class UiHttpSecurityContext:
    csrf_token: str
    allow_non_loopback: bool
    externally_supplied_token: bool
    bind_host: str
    cookie_name: str = "asa_controller_token"

    @property
    def local_only(self) -> bool:
        return _is_loopback_host(self.bind_host)


def build_security_context(
    bind_host: str,
    *,
    allow_non_loopback: bool = False,
    csrf_token: str | None = None,
) -> UiHttpSecurityContext:
    env_token = os.environ.get(CONTROLLER_TOKEN_ENV)
    token = csrf_token or env_token or secrets.token_urlsafe(32)
    return UiHttpSecurityContext(
        csrf_token=token,
        allow_non_loopback=allow_non_loopback,
        externally_supplied_token=bool(csrf_token or env_token),
        bind_host=bind_host,
    )


def require_safe_bind(context: UiHttpSecurityContext) -> None:
    if context.local_only:
        return
    if not context.externally_supplied_token:
        raise PermissionError("non_loopback_bind_requires_controller_token")
    if not context.allow_non_loopback:
        raise PermissionError("non_loopback_bind_requires_explicit_opt_in")


def authorized_state_change(headers: HeaderMapping, context: UiHttpSecurityContext) -> bool:
    if context.allow_non_loopback and not context.local_only:
        return _bearer_token(headers) == context.csrf_token
    if not request_origin_allowed(headers):
        return False
    return (
        _header_value(headers, "X-ASA-CSRF-Token") == context.csrf_token
        or _bearer_token(headers) == context.csrf_token
        or _cookie_token(headers, context.cookie_name) == context.csrf_token
    )


def security_payload(context: UiHttpSecurityContext) -> JsonMap:
    external = context.allow_non_loopback and not context.local_only
    return {
        "state_changing_post_auth": "external_bearer" if external else "csrf_or_bearer",
        "token_exposed": False,
        "token_transport": "external_bearer" if external else "http_only_cookie_or_header",
        "non_loopback_bind_allowed": context.allow_non_loopback,
    }


def session_cookie_header(context: UiHttpSecurityContext) -> str | None:
    if context.allow_non_loopback and not context.local_only:
        return None
    cookie = SimpleCookie()
    cookie[context.cookie_name] = context.csrf_token
    cookie[context.cookie_name]["httponly"] = True
    cookie[context.cookie_name]["samesite"] = "Strict"
    cookie[context.cookie_name]["path"] = "/"
    return cookie.output(header="").strip()


def request_origin_allowed(headers: HeaderMapping) -> bool:
    origin = _header_value(headers, "Origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    host = parsed.hostname
    return bool(host and _is_loopback_host(host))


def _bearer_token(headers: HeaderMapping) -> str | None:
    authorization = _header_value(headers, "Authorization")
    if not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    return token or None


def _cookie_token(headers: HeaderMapping, cookie_name: str) -> str | None:
    raw_cookie = _header_value(headers, "Cookie")
    if not raw_cookie:
        return None
    cookie = SimpleCookie(raw_cookie)
    morsel = cookie.get(cookie_name)
    if morsel is None:
        return None
    return morsel.value


def _header_value(headers: HeaderMapping, name: str) -> str:
    value = headers.get(name, "")
    return value if isinstance(value, str) else ""


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"", "localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False
