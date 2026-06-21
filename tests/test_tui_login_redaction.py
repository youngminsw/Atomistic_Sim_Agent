from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

from sim_agent.cli.tui_browser_oauth import start_browser_oauth
from sim_agent.cli.tui_login import handle_login
from sim_agent.cli.tui_login_profiles import LoginTarget
from sim_agent.cli.tui_state import initial_state
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV


class FakeTty(StringIO):
    def isatty(self) -> bool:
        return True


def test_tty_login_token_output_is_redacted_and_recommends_model_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a normal TTY login using pasted credentials and an explicit store.
    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    output = FakeTty()

    # When: login succeeds through the TUI login command path.
    handle_login(
        (
            "api-key",
            "--provider",
            "openai",
            "--profile",
            "openai_api",
            "--api-key",
            "tty-secret-token",
            "--credential-store",
            str(store),
        ),
        initial_state(tmp_path / "session"),
        output,
    )

    # Then: output is friendly, redacted, and points to the next model step.
    text = output.getvalue()
    assert "Login successful." in text
    assert "Signed in with OpenAI API." in text
    assert "Next: /model profile codex-pro" in text
    assert "login_ok=true" not in text
    assert "credential_store=" not in text
    assert str(store) not in text
    assert "access_token" not in text
    assert "tty-secret-token" not in text
    assert "refresh_token" not in text


def test_non_tty_login_token_output_keeps_machine_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: non-TTY automation logs in with a pasted token.
    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    output = StringIO()

    # When: login succeeds through the same command path.
    handle_login(
        (
            "api-key",
            "--provider",
            "openai",
            "--profile",
            "openai_api",
            "--api-key",
            "machine-secret-token",
            "--credential-store",
            str(store),
        ),
        initial_state(tmp_path / "session"),
        output,
    )

    # Then: existing machine-readable keys remain available for scripts.
    text = output.getvalue()
    assert "login_ok=true" in text
    assert "provider=openai auth_mode=api_key" in text
    assert f"credential_store={store}" in text
    assert "machine-secret-token" not in text


def test_tty_configured_browser_oauth_keeps_copyable_url_without_store_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a normal TTY starts a configured OAuth gateway fallback.
    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    monkeypatch.setenv("ASA_BROWSER_OAUTH_OPEN", "0")
    output = FakeTty()

    # When: browser OAuth starts.
    start_browser_oauth(
        LoginTarget("chatgpt_codex", "openai-codex", "oauth", "ChatGPT Plus/Pro", "OpenAI"),
        (
            "--auth-url",
            "https://auth.example.test/start/{provider}/{profile}",
            "--credential-store",
            str(store),
            "--no-open",
        ),
        initial_state(tmp_path / "session"),
        output,
    )

    # Then: the URL remains copyable, but debug keys and store paths are hidden.
    text = output.getvalue()
    assert "Open this URL in your browser:" in text
    assert "https://auth.example.test/start/openai-codex/chatgpt_codex" in text
    assert "browser_oauth_url=" not in text
    assert "credential_store=" not in text
    assert str(store) not in text
    assert "access_token" not in text
    assert "refresh_token" not in text


def test_kimi_tty_oauth_success_is_redacted(tmp_path: Path, monkeypatch) -> None:
    # Given: Kimi device OAuth returns access and refresh tokens.
    class KimiHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/api/oauth/device_authorization":
                payload = {
                    "user_code": "KIMI-CODE",
                    "device_code": "kimi-device-code",
                    "verification_uri": f"http://127.0.0.1:{self.server.server_port}/device",
                    "verification_uri_complete": f"http://127.0.0.1:{self.server.server_port}/device?user_code=KIMI-CODE",
                    "expires_in": 30,
                    "interval": 1,
                }
            else:
                payload = {
                    "access_token": "kimi-access-token",
                    "refresh_token": "kimi-refresh-token",
                    "expires_in": 3600,
                }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), KimiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    output = FakeTty()

    # When: device OAuth completes in a normal TTY.
    try:
        start_browser_oauth(
            LoginTarget("kimi_code", "kimi-code", "oauth", "Kimi Code", "Moonshot / Kimi"),
            (
                "--oauth-host",
                f"http://127.0.0.1:{server.server_port}",
                "--device-timeout-s",
                "5",
                "--no-open",
            ),
            initial_state(tmp_path / "session"),
            output,
        )
    finally:
        server.shutdown()
        server.server_close()

    # Then: success is friendly and does not leak token or credential internals.
    text = output.getvalue()
    assert "Login successful." in text
    assert "Signed in with Kimi Code." in text
    assert "Next: /model profile codex-pro" in text
    assert "browser_oauth_credential_saved=true" not in text
    assert "login_ok=true" not in text
    assert "credential_store=" not in text
    assert str(store) not in text
    assert "access_token" not in text
    assert "kimi-access-token" not in text
    assert "refresh_token" not in text
    assert "kimi-refresh-token" not in text
