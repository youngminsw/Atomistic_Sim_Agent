from __future__ import annotations

import json
import os
import platform
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_provider

from .tui_browser_launch import open_url_in_browser, write_oauth_browser_block
from .tui_login_profiles import LoginTarget
from .tui_parse import parse_options
from .tui_redaction import machine_output, write_login_success
from .tui_state import TuiState, append_event


CLIENT_ID: Final = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST: Final = "https://auth.kimi.com"
DEVICE_AUTHORIZATION_PATH: Final = "/api/oauth/device_authorization"
TOKEN_PATH: Final = "/api/oauth/token"
DEVICE_ID_FILENAME: Final = "kimi-device-id"


@dataclass(frozen=True, slots=True)
class KimiDeviceAuthorization:
    user_code: str
    device_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in_s: int
    interval_s: float


@dataclass(frozen=True, slots=True)
class KimiToken:
    access_token: str
    refresh_token: str
    expires_in_s: int


@dataclass(frozen=True, slots=True)
class KimiOAuthOutcome:
    provider: str
    profile: str
    started: bool
    opened: bool
    url: str | None = None
    blocker: str | None = None


def start_kimi_device_oauth(
    target: LoginTarget,
    args: tuple[str, ...],
    state: TuiState,
    output_stream: TextIO,
) -> KimiOAuthOutcome:
    parsed = parse_options(args)
    credential_store = parsed.options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store
    host = _oauth_host(parsed.options)
    timeout_s = _float_option(parsed.options, "device_timeout_s", 900.0)
    try:
        device = _request_device_authorization(host, timeout_s=15.0)
    except KimiOAuthError as exc:
        blocker = f"device_authorization_failed:{exc.code}"
        _write_start_blocked(target, output_stream, blocker)
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return KimiOAuthOutcome(target.provider, target.profile, False, False, blocker=blocker)

    opened = _open_browser(device.verification_uri_complete, parsed.flags)
    write_oauth_browser_block(
        output_stream,
        url=device.verification_uri_complete,
        opened=opened,
        user_code=device.user_code,
        instructions="Enter the code in the browser to finish Kimi login.",
    )
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=true\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_url={device.verification_uri_complete}\n")
        output_stream.write(f"browser_oauth_user_code={device.user_code}\n")
        output_stream.write(f"browser_opened={opened}\n")
        output_stream.write("browser_oauth_waiting_for_device_authorization=true\n")
    else:
        output_stream.write("Waiting for browser login to finish...\n")
    append_event(state, "browser_oauth_started", f"provider={target.provider} kimi_device_code")

    deadline = time.monotonic() + min(timeout_s, float(device.expires_in_s))
    interval_s = max(1.0, device.interval_s)
    while time.monotonic() < deadline:
        time.sleep(interval_s)
        try:
            token = _poll_token(host, device.device_code, timeout_s=15.0)
        except KimiOAuthError as exc:
            if exc.code == "authorization_pending":
                continue
            if exc.code == "slow_down":
                interval_s += 5.0
                continue
            blocker = f"device_poll_failed:{exc.code}"
            _write_oauth_blocker(output_stream, blocker)
            append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
            return KimiOAuthOutcome(target.provider, target.profile, True, opened, device.verification_uri_complete, blocker)
        _save_token(target, token, output_stream, state)
        return KimiOAuthOutcome(target.provider, target.profile, True, opened, device.verification_uri_complete)

    blocker = "device_authorization_timeout"
    _write_oauth_blocker(output_stream, blocker)
    append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
    return KimiOAuthOutcome(target.provider, target.profile, True, opened, device.verification_uri_complete, blocker)


class KimiOAuthError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _request_device_authorization(host: str, *, timeout_s: float) -> KimiDeviceAuthorization:
    payload = _post_form(
        f"{host}{DEVICE_AUTHORIZATION_PATH}",
        {"client_id": CLIENT_ID},
        timeout_s=timeout_s,
    )
    user_code = _required_text(payload, "user_code")
    device_code = _required_text(payload, "device_code")
    verification_uri = _required_text(payload, "verification_uri")
    verification_uri_complete = _optional_text(payload, "verification_uri_complete") or verification_uri
    return KimiDeviceAuthorization(
        user_code=user_code,
        device_code=device_code,
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        expires_in_s=_int_from_value(payload.get("expires_in"), 900),
        interval_s=_float_from_value(payload.get("interval"), 5.0),
    )


def _poll_token(host: str, device_code: str, *, timeout_s: float) -> KimiToken:
    payload = _post_form(
        f"{host}{TOKEN_PATH}",
        {
            "client_id": CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
        timeout_s=timeout_s,
    )
    error = _optional_text(payload, "error")
    if error:
        raise KimiOAuthError(error)
    access = _required_text(payload, "access_token")
    refresh = _optional_text(payload, "refresh_token") or access
    expires = _int_from_value(payload.get("expires_in"), 3600)
    return KimiToken(access, refresh, expires)


def _post_form(url: str, form: dict[str, str], *, timeout_s: float) -> dict[str, object]:
    body = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            **_kimi_headers(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise KimiOAuthError(str(exc.code)) from exc
    except urllib.error.URLError as exc:
        code = exc.reason if isinstance(exc.reason, str) else "network_error"
        raise KimiOAuthError(code) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KimiOAuthError("invalid_json") from exc
    if not isinstance(value, dict):
        raise KimiOAuthError("json_object_required")
    return value


def _save_token(target: LoginTarget, token: KimiToken, output_stream: TextIO, state: TuiState) -> None:
    payload = login_model_provider(
        {
            "provider": target.provider,
            "login_profile": target.profile,
            "access_token": token.access_token,
            "refresh_token": token.refresh_token,
            "auth_mode": "oauth",
            "expires_in_s": token.expires_in_s,
        }
    )
    append_event(state, "model_login", f"provider={payload['provider']} auth_mode=oauth")
    if machine_output(output_stream):
        output_stream.write("browser_oauth_credential_saved=true\n")
        output_stream.write("login_ok=true\n")
        output_stream.write(f"provider={payload['provider']} auth_mode=oauth\n")
        output_stream.write(f"login_profile={target.profile}\n")
        output_stream.write(f"provider_credential_store={payload['provider_credential_store']}\n")
        return
    write_login_success(output_stream, provider=payload["provider"], label=target.label)


def _oauth_host(options: dict[str, str]) -> str:
    value = (
        options.get("oauth_host")
        or os.environ.get("KIMI_CODE_OAUTH_HOST")
        or os.environ.get("KIMI_OAUTH_HOST")
        or DEFAULT_OAUTH_HOST
    )
    return value.rstrip("/")


def _kimi_headers() -> dict[str, str]:
    return {
        "User-Agent": "ASA/0",
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": "0",
        "X-Msh-Device-Name": platform.node() or "asa",
        "X-Msh-Device-Model": f"{platform.system()} {platform.release()} {platform.machine()}".strip(),
        "X-Msh-Os-Version": platform.version(),
        "X-Msh-Device-Id": _device_id(),
    }


def _device_id() -> str:
    root = Path.home() / ".asa"
    path = root / DEVICE_ID_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    if existing:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    generated = uuid.uuid4().hex
    path.write_text(generated + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return generated


def _required_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    raise KimiOAuthError(f"missing_{field}")


def _optional_text(payload: dict[str, object], field: str) -> str | None:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return None


def _int_from_value(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return default


def _float_from_value(value: object, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _float_option(options: dict[str, str], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _open_browser(url: str, flags: tuple[str, ...]) -> bool:
    return open_url_in_browser(url, flags)


def _write_start_blocked(target: LoginTarget, output_stream: TextIO, blocker: str) -> None:
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=false\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_blocker={blocker}\n")
        return
    output_stream.write(f"Kimi login could not start: {blocker}\n")


def _write_oauth_blocker(output_stream: TextIO, blocker: str) -> None:
    if machine_output(output_stream):
        output_stream.write(f"browser_oauth_blocker={blocker}\n")
        output_stream.write("browser_oauth_credential_saved=false\n")
        return
    output_stream.write(f"Login did not finish: {blocker}\n")
