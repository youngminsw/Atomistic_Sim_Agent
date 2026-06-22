from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TextIO

from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_gateway

from .tui_browser_launch import open_url_in_browser, write_oauth_browser_block
from .tui_login_profiles import LoginTarget
from .tui_parse import parse_options
from .tui_redaction import machine_output, write_login_success
from .tui_state import TuiState, append_event


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
SCOPE = "openid profile email offline_access"
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_AUTH_URL = "https://auth.openai.com/codex/device"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"


@dataclass(frozen=True, slots=True)
class OAuthFlowOutcome:
    provider: str
    profile: str
    started: bool
    opened: bool
    url: str | None = None
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class CallbackResult:
    code: str
    state: str


@dataclass(frozen=True, slots=True)
class TokenResult:
    access_token: str
    refresh_token: str
    expires_in_s: int


def start_openai_codex_oauth(
    target: LoginTarget,
    args: tuple[str, ...],
    state: TuiState,
    output_stream: TextIO,
) -> OAuthFlowOutcome:
    if target.profile == "chatgpt_codex_device":
        return _start_device_flow(target, args, state, output_stream)
    return _start_browser_flow(target, args, state, output_stream)


def _start_browser_flow(
    target: LoginTarget,
    args: tuple[str, ...],
    state: TuiState,
    output_stream: TextIO,
) -> OAuthFlowOutcome:
    parsed = parse_options(args)
    _configure_credential_store(parsed.options)
    port = _int_option(parsed.options, "callback_port", CALLBACK_PORT)
    timeout_s = _float_option(parsed.options, "callback_timeout_s", _float_env("ASA_OAUTH_CALLBACK_TIMEOUT_S", 300.0))
    token_url = parsed.options.get("token_url") or os.environ.get("ASA_OPENAI_CODEX_TOKEN_URL") or TOKEN_URL
    client_id = parsed.options.get("client_id") or os.environ.get("ASA_OPENAI_CODEX_CLIENT_ID") or CLIENT_ID
    auth_url_base = parsed.options.get("authorize_url") or os.environ.get("ASA_OPENAI_CODEX_AUTHORIZE_URL") or AUTHORIZE_URL
    pkce_verifier, pkce_challenge = _pkce_pair()
    csrf_state = secrets.token_urlsafe(24)

    try:
        server = _callback_server(port, csrf_state)
    except OSError as exc:
        blocker = "callback_port_unavailable"
        _write_openai_start_blocked(target, output_stream, blocker, str(exc))
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return OAuthFlowOutcome(target.provider, target.profile, False, False, blocker=blocker)

    redirect_uri = _redirect_uri(server.server_port, port)
    auth_url = _auth_url(auth_url_base, client_id, redirect_uri, csrf_state, pkce_challenge)
    opened = _open_browser(auth_url, parsed.flags)
    write_oauth_browser_block(
        output_stream,
        url=auth_url,
        opened=opened,
        callback_url=redirect_uri,
        instructions="A browser window should open. Complete login to finish.",
    )
    _write_oauth_started(target, output_stream, auth_url, opened, callback_url=redirect_uri)
    append_event(state, "browser_oauth_started", f"provider={target.provider} callback={redirect_uri}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        callback = server.callback_queue.get(timeout=timeout_s)
    except queue.Empty:
        server.shutdown()
        blocker = "callback_timeout"
        _write_oauth_blocker(output_stream, blocker)
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return OAuthFlowOutcome(target.provider, target.profile, True, opened, auth_url, blocker)
    finally:
        server.shutdown()
        server.server_close()

    if isinstance(callback, str):
        blocker = callback
        _write_oauth_blocker(output_stream, blocker)
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return OAuthFlowOutcome(target.provider, target.profile, True, opened, auth_url, blocker)

    try:
        token = _exchange_code_for_token(token_url, client_id, callback.code, redirect_uri, pkce_verifier)
    except OAuthTokenError as exc:
        blocker = f"token_exchange_failed:{exc.code}"
        _write_oauth_blocker(output_stream, blocker)
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return OAuthFlowOutcome(target.provider, target.profile, True, opened, auth_url, blocker)

    _save_token(target, token, output_stream, state)
    return OAuthFlowOutcome(target.provider, target.profile, True, opened, auth_url)


def _start_device_flow(
    target: LoginTarget,
    args: tuple[str, ...],
    state: TuiState,
    output_stream: TextIO,
) -> OAuthFlowOutcome:
    parsed = parse_options(args)
    _configure_credential_store(parsed.options)
    usercode_url = parsed.options.get("device_usercode_url") or os.environ.get("ASA_OPENAI_CODEX_DEVICE_USERCODE_URL") or DEVICE_USERCODE_URL
    device_token_url = parsed.options.get("device_token_url") or os.environ.get("ASA_OPENAI_CODEX_DEVICE_TOKEN_URL") or DEVICE_TOKEN_URL
    token_url = parsed.options.get("token_url") or os.environ.get("ASA_OPENAI_CODEX_TOKEN_URL") or TOKEN_URL
    timeout_s = _float_option(parsed.options, "device_timeout_s", _float_env("ASA_OAUTH_DEVICE_TIMEOUT_S", 600.0))
    client_id = parsed.options.get("client_id") or os.environ.get("ASA_OPENAI_CODEX_CLIENT_ID") or CLIENT_ID
    try:
        init = _post_json(usercode_url, {"client_id": client_id}, timeout_s=15.0)
        device_auth_id = _required_text(init, "device_auth_id")
        user_code = _required_text(init, "user_code")
        interval_s = _float_from_value(init.get("interval"), 5.0)
    except OAuthTokenError as exc:
        blocker = f"device_authorization_failed:{exc.code}"
        _write_openai_start_blocked(target, output_stream, blocker, "")
        append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
        return OAuthFlowOutcome(target.provider, target.profile, False, False, blocker=blocker)

    opened = _open_browser(DEVICE_AUTH_URL, parsed.flags)
    write_oauth_browser_block(
        output_stream,
        url=DEVICE_AUTH_URL,
        opened=opened,
        user_code=user_code,
        instructions="Enter the code in the browser to finish ChatGPT login.",
    )
    _write_oauth_started(target, output_stream, DEVICE_AUTH_URL, opened, user_code=user_code)
    append_event(state, "browser_oauth_started", f"provider={target.provider} device_code")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(max(1.0, interval_s))
        try:
            payload = _post_json(device_token_url, {"device_auth_id": device_auth_id, "user_code": user_code}, timeout_s=15.0)
        except OAuthTokenError as exc:
            if exc.code in {"403", "404"}:
                continue
            blocker = f"device_poll_failed:{exc.code}"
            _write_oauth_blocker(output_stream, blocker)
            append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
            return OAuthFlowOutcome(target.provider, target.profile, True, opened, DEVICE_AUTH_URL, blocker)
        authorization_code = _required_text(payload, "authorization_code")
        code_verifier = _required_text(payload, "code_verifier")
        try:
            token = _exchange_code_for_token(token_url, client_id, authorization_code, DEVICE_REDIRECT_URI, code_verifier)
        except OAuthTokenError as exc:
            blocker = f"token_exchange_failed:{exc.code}"
            _write_oauth_blocker(output_stream, blocker)
            append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
            return OAuthFlowOutcome(target.provider, target.profile, True, opened, DEVICE_AUTH_URL, blocker)
        _save_token(target, token, output_stream, state)
        return OAuthFlowOutcome(target.provider, target.profile, True, opened, DEVICE_AUTH_URL)

    blocker = "device_authorization_timeout"
    _write_oauth_blocker(output_stream, blocker)
    append_event(state, "browser_oauth_blocked", f"provider={target.provider} {blocker}")
    return OAuthFlowOutcome(target.provider, target.profile, True, opened, DEVICE_AUTH_URL, blocker)


class OAuthTokenError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _CallbackServer(ThreadingHTTPServer):
    callback_queue: queue.Queue[CallbackResult | str]
    callback_path: str
    expected_state: str


def _callback_server(port: int, expected_state: str) -> _CallbackServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path != self.server.callback_path:
                self.send_response(404)
                self.end_headers()
                return
            error = _first(query, "error")
            if error:
                self._finish(500, f"authorization_failed:{error}")
                return
            code = _first(query, "code")
            returned_state = _first(query, "state") or ""
            if not code:
                self._finish(500, "missing_authorization_code")
                return
            if returned_state != self.server.expected_state:
                self._finish(500, "state_mismatch")
                return
            self.server.callback_queue.put(CallbackResult(code, returned_state))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>ASA login complete</h1>You can return to the terminal.</body></html>")

        def _finish(self, status: int, message: str) -> None:
            self.server.callback_queue.put(message)
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(message.encode("utf-8"))

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = _CallbackServer(("127.0.0.1", port), Handler)
    server.callback_queue = queue.Queue(maxsize=1)
    server.callback_path = CALLBACK_PATH
    server.expected_state = expected_state
    return server


def _redirect_uri(actual_port: int, requested_port: int) -> str:
    port = requested_port if requested_port != 0 else actual_port
    return f"http://localhost:{port}{CALLBACK_PATH}"


def _auth_url(base_url: str, client_id: str, redirect_uri: str, csrf_state: str, pkce_challenge: str) -> str:
    params = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "code_challenge": pkce_challenge,
            "code_challenge_method": "S256",
            "state": csrf_state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "asa",
        }
    )
    return f"{base_url}?{params}"


def _exchange_code_for_token(token_url: str, client_id: str, code: str, redirect_uri: str, verifier: str) -> TokenResult:
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        }
    ).encode("utf-8")
    payload = _post_form(token_url, body, timeout_s=15.0)
    access = _required_text(payload, "access_token")
    refresh = _required_text(payload, "refresh_token")
    expires = _int_from_value(payload.get("expires_in"), 3600)
    return TokenResult(access, refresh, expires)


def _post_form(url: str, body: bytes, *, timeout_s: float) -> dict[str, object]:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    return _open_json(request, timeout_s)


def _post_json(url: str, payload: dict[str, str], *, timeout_s: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return _open_json(request, timeout_s)


def _open_json(request: urllib.request.Request, timeout_s: float) -> dict[str, object]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise OAuthTokenError(str(exc.code)) from exc
    except urllib.error.URLError as exc:
        raise OAuthTokenError(exc.reason if isinstance(exc.reason, str) else "network_error") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OAuthTokenError("invalid_json") from exc
    if not isinstance(value, dict):
        raise OAuthTokenError("json_object_required")
    return value


def _save_token(target: LoginTarget, token: TokenResult, output_stream: TextIO, state: TuiState) -> None:
    payload = login_model_gateway(
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
        output_stream.write(f"credential_store={payload['credential_store']}\n")
        return
    write_login_success(output_stream, provider=payload["provider"], label=target.label)


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _required_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    raise OAuthTokenError(f"missing_{field}")


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


def _int_option(options: dict[str, str], key: str, default: int) -> int:
    value = options.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_option(options: dict[str, str], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def _configure_credential_store(options: dict[str, str]) -> None:
    credential_store = options.get("credential_store")
    if credential_store is not None:
        os.environ[CREDENTIAL_STORE_ENV] = credential_store


def _open_browser(url: str, flags: tuple[str, ...]) -> bool:
    return open_url_in_browser(url, flags)


def _write_openai_start_blocked(target: LoginTarget, output_stream: TextIO, blocker: str, detail: str) -> None:
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=false\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_blocker={blocker}\n")
        if detail:
            output_stream.write(f"browser_oauth_detail={detail}\n")
        return
    output_stream.write(f"OAuth login could not start: {blocker}\n")
    if detail:
        output_stream.write(f"{detail}\n")


def _write_oauth_started(
    target: LoginTarget,
    output_stream: TextIO,
    url: str,
    opened: bool,
    *,
    callback_url: str | None = None,
    user_code: str | None = None,
) -> None:
    if machine_output(output_stream):
        output_stream.write("browser_oauth_started=true\n")
        output_stream.write(f"browser_oauth_provider={target.provider}\n")
        output_stream.write(f"browser_oauth_profile={target.profile}\n")
        output_stream.write(f"browser_oauth_url={url}\n")
        if callback_url is not None:
            output_stream.write(f"browser_oauth_callback={callback_url}\n")
        if user_code is not None:
            output_stream.write(f"browser_oauth_user_code={user_code}\n")
        output_stream.write(f"browser_opened={opened}\n")
        wait_key = "device_authorization" if user_code is not None else "callback"
        output_stream.write(f"browser_oauth_waiting_for_{wait_key}=true\n")
        return
    output_stream.write("Waiting for browser login to finish...\n")


def _write_oauth_blocker(output_stream: TextIO, blocker: str) -> None:
    if machine_output(output_stream):
        output_stream.write(f"browser_oauth_blocker={blocker}\n")
        output_stream.write("browser_oauth_credential_saved=false\n")
        return
    output_stream.write(f"Login did not finish: {blocker}\n")
