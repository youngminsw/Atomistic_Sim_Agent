from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

from sim_agent.runtime_config import default_runtime_config, save_runtime_config


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_non_tty_cancel_after_login_help_does_not_start_run() -> None:
    result = _run_module_interactive(["/login", "cancel", "/exit"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Login Options" in result.stdout
    assert "cancelled=true" in result.stdout
    assert "run_prepared=true" not in result.stdout
    assert "agent_run_ledger_path=" not in result.stdout


def test_interactive_login_selector_stores_credentials_without_endpoint_change(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.cli.tui_login import LoginTarget, handle_login
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.runtime_config import RUNTIME_CONFIG_ENV
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    class FakeSelector:
        def choose_target(self, default: str) -> LoginTarget:
            assert default == "openai-codex"
            return LoginTarget(
                profile="openai_api",
                provider="openai",
                token_mode="api_key",
                label="OpenAI API",
                company="OpenAI",
            )

        def prompt_token(self, target: LoginTarget) -> str:
            assert target.token_mode == "api_key"
            return "interactive-secret-token"

    store = tmp_path / "credentials.json"
    runtime_config = tmp_path / "runtime-config.json"
    save_runtime_config(default_runtime_config(), runtime_config)
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    monkeypatch.setenv(RUNTIME_CONFIG_ENV, str(runtime_config))
    output = StringIO()

    handle_login((), initial_state(tmp_path), output, FakeSelector())

    assert "login_ok=true" in output.getvalue()
    assert "provider=openai auth_mode=api_key" in output.getvalue()
    assert "model_endpoint_saved=true" not in output.getvalue()
    assert "interactive-secret-token" not in output.getvalue()
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["openai"]["provider"] == "openai"
    saved = json.loads(runtime_config.read_text(encoding="utf-8"))
    assert saved["model_endpoint"]["provider"] == "openai-codex"
    assert saved["model_endpoint"]["model"] == "gpt-5-codex"


def test_interactive_oauth_selector_starts_browser_flow_without_token_prompt(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.cli.tui_login import LoginTarget, handle_login
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    class FakeSelector:
        def choose_target(self, default: str) -> LoginTarget:
            assert default == "openai-codex"
            return LoginTarget(
                profile="chatgpt_codex",
                provider="openai-codex",
                token_mode="oauth",
                label="ChatGPT Plus/Pro",
                company="OpenAI",
            )

        def prompt_token(self, target: LoginTarget) -> str:
            raise AssertionError("OAuth login must not ask for pasted tokens")

    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    monkeypatch.setenv("ASA_BROWSER_OAUTH_OPEN", "0")
    monkeypatch.setenv(
        "ASA_OAUTH_AUTH_URL_CHATGPT_CODEX",
        "https://auth.example.test/start/{provider}/{profile}",
    )
    output = StringIO()

    handle_login((), initial_state(tmp_path), output, FakeSelector())

    text = output.getvalue()
    assert "browser_oauth_started=true" in text
    assert "browser_oauth_provider=openai-codex" in text
    assert "browser_oauth_profile=chatgpt_codex" in text
    assert "Open this URL in your browser:" in text
    assert "browser_oauth_url=https://auth.example.test/start/openai-codex/chatgpt_codex" in text
    assert "browser_opened=False" in text
    assert "login_ok=true" not in text
    assert not store.exists()


def test_noninteractive_openai_codex_oauth_starts_and_times_out_without_callback() -> None:
    result = _run_module_interactive(
        ["/login oauth --provider openai-codex --callback-port 0 --callback-timeout-s 0.01 --no-open", "/exit"]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "browser_oauth_started=true" in result.stdout
    assert "browser_oauth_provider=openai-codex" in result.stdout
    assert "browser_oauth_profile=chatgpt_codex" in result.stdout
    assert "Open this URL in your browser:" in result.stdout
    assert "browser_oauth_url=https://auth.openai.com/oauth/authorize?" in result.stdout
    assert "client_id=app_EMoamEEZ73f0CkXaXp7hrann" in result.stdout
    assert "browser_oauth_blocker=callback_timeout" in result.stdout
    assert "login_ok=true" not in result.stdout


def test_noninteractive_oauth_with_auth_url_launches_browser_receipt() -> None:
    result = _run_module_interactive(
        [
            "/login oauth --provider openai-codex --auth-url https://auth.example.test/start/{provider}/{profile} --no-open",
            "/exit",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "browser_oauth_started=true" in result.stdout
    assert "browser_oauth_provider=openai-codex" in result.stdout
    assert "browser_oauth_profile=chatgpt_codex" in result.stdout
    assert "Open this URL in your browser:" in result.stdout
    assert "browser_oauth_url=https://auth.example.test/start/openai-codex/chatgpt_codex" in result.stdout
    assert "browser_opened=False" in result.stdout
    assert "login_ok=true" not in result.stdout


def test_login_profile_selects_company_provider_without_model_coupling() -> None:
    from sim_agent.cli.tui_login_profiles import choose_login_target

    class FakeStream(StringIO):
        def isatty(self) -> bool:
            return False

    input_stream = FakeStream("3\n1\n")
    output = StringIO()

    target = choose_login_target("openai-codex", input_stream, output)

    assert target is not None
    assert target.profile == "google_gemini_cli"
    assert target.provider == "google-gemini-cli"
    assert target.company == "Google"
    assert "Google Cloud Code Assist" in output.getvalue()
    assert "Login Company" in output.getvalue()
    assert "Login Provider" in output.getvalue()
    assert "gemini-3-pro-preview" not in output.getvalue()


def test_model_use_accepts_source_provider_reference_and_saves_runtime_provider(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/model use openai-codex/gpt-5.3-codex-spark --reasoning-effort high",
            "/model status",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_saved=true" in result.stdout
    assert "provider=openai-codex model=gpt-5.3-codex-spark reasoning_effort=high" in result.stdout
    assert saved["model_endpoint"] == {
        "api_key_env": "MODEL_GATEWAY_TOKEN",
        "auth_mode": "gateway",
        "base_url": "https://model-gateway.local/v1",
        "model": "gpt-5.3-codex-spark",
        "provider": "openai-codex",
        "reasoning_effort": "high",
    }


def test_openai_codex_browser_oauth_callback_stores_credential(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.cli import tui_browser_launch
    from sim_agent.cli.tui_browser_oauth import start_browser_oauth
    from sim_agent.cli.tui_login_profiles import LoginTarget
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    token_requests: list[dict[str, list[str]]] = []

    class TokenHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            token_requests.append(urllib.parse.parse_qs(body))
            payload = {
                "access_token": "fake-access-token",
                "refresh_token": "fake-refresh-token",
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

    token_server = ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
    token_thread = threading.Thread(target=token_server.serve_forever, daemon=True)
    token_thread.start()

    def fake_open(url: str, *_args: object, **_kwargs: object) -> bool:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query["redirect_uri"][0]
        state = query["state"][0]

        def complete_callback() -> None:
            time.sleep(0.05)
            callback_url = f"{redirect_uri}?code=fake-code&state={state}"
            with urllib.request.urlopen(callback_url, timeout=5) as response:
                response.read()

        threading.Thread(target=complete_callback, daemon=True).start()
        return True

    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    monkeypatch.setattr(tui_browser_launch, "_is_wsl", lambda: False)
    monkeypatch.setattr(tui_browser_launch.webbrowser, "open", fake_open)
    output = StringIO()

    try:
        start_browser_oauth(
            LoginTarget("chatgpt_codex", "openai-codex", "oauth", "ChatGPT Plus/Pro", "OpenAI"),
            (
                "--callback-port",
                "0",
                "--callback-timeout-s",
                "5",
                "--token-url",
                f"http://127.0.0.1:{token_server.server_port}/token",
            ),
            initial_state(tmp_path / "session"),
            output,
        )
    finally:
        token_server.shutdown()
        token_server.server_close()

    text = output.getvalue()
    assert "browser_oauth_started=true" in text
    assert "Open this URL in your browser:" in text
    assert "browser_oauth_credential_saved=true" in text
    assert "login_ok=true" in text
    assert "provider=openai-codex auth_mode=oauth" in text
    assert token_requests[0]["code"] == ["fake-code"]
    assert token_requests[0]["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
    assert token_requests[0]["grant_type"] == ["authorization_code"]
    assert token_requests[0]["code_verifier"][0]
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["openai-codex"]["credentials"]["access"] == "fake-access-token"
    assert payload["openai-codex"]["loginProfile"] == "chatgpt_codex"


def test_openai_codex_oauth_tty_success_uses_friendly_output(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.cli import tui_browser_launch
    from sim_agent.cli.tui_browser_oauth import start_browser_oauth
    from sim_agent.cli.tui_login_profiles import LoginTarget
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    class FakeTty(StringIO):
        def isatty(self) -> bool:
            return True

    class TokenHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = {
                "access_token": "fake-access-token",
                "refresh_token": "fake-refresh-token",
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

    token_server = ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
    token_thread = threading.Thread(target=token_server.serve_forever, daemon=True)
    token_thread.start()

    def fake_open(url: str, *_args: object, **_kwargs: object) -> bool:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query["redirect_uri"][0]
        state = query["state"][0]

        def complete_callback() -> None:
            time.sleep(0.05)
            callback_url = f"{redirect_uri}?code=fake-code&state={state}"
            with urllib.request.urlopen(callback_url, timeout=5) as response:
                response.read()

        threading.Thread(target=complete_callback, daemon=True).start()
        return True

    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    monkeypatch.setattr(tui_browser_launch, "_is_wsl", lambda: False)
    monkeypatch.setattr(tui_browser_launch.webbrowser, "open", fake_open)
    output = FakeTty()

    try:
        start_browser_oauth(
            LoginTarget("chatgpt_codex", "openai-codex", "oauth", "ChatGPT Plus/Pro", "OpenAI"),
            (
                "--callback-port",
                "0",
                "--callback-timeout-s",
                "5",
                "--token-url",
                f"http://127.0.0.1:{token_server.server_port}/token",
            ),
            initial_state(tmp_path / "session"),
            output,
        )
    finally:
        token_server.shutdown()
        token_server.server_close()

    text = output.getvalue()
    assert "Login successful." in text
    assert "Signed in with ChatGPT Plus/Pro." in text
    assert "Credential saved for openai-codex." in text
    assert "browser_oauth_credential_saved=true" not in text
    assert "login_ok=true" not in text


def test_kimi_device_oauth_stores_credential(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.cli.tui_browser_oauth import start_browser_oauth
    from sim_agent.cli.tui_login_profiles import LoginTarget
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    requests: list[tuple[str, dict[str, list[str]]]] = []

    class KimiHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            form = urllib.parse.parse_qs(body)
            requests.append((self.path, form))
            if self.path == "/api/oauth/device_authorization":
                payload = {
                    "user_code": "KIMI-CODE",
                    "device_code": "kimi-device-code",
                    "verification_uri": f"http://127.0.0.1:{self.server.server_port}/device",
                    "verification_uri_complete": f"http://127.0.0.1:{self.server.server_port}/device?user_code=KIMI-CODE",
                    "expires_in": 30,
                    "interval": 1,
                }
            elif self.path == "/api/oauth/token":
                payload = {
                    "access_token": "kimi-access-token",
                    "refresh_token": "kimi-refresh-token",
                    "expires_in": 3600,
                }
            else:
                self.send_response(404)
                self.end_headers()
                return
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
    output = StringIO()

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

    text = output.getvalue()
    assert "browser_oauth_started=true" in text
    assert "browser_oauth_provider=kimi-code" in text
    assert "Open this URL in your browser:" in text
    assert "browser_oauth_user_code=KIMI-CODE" in text
    assert "browser_oauth_credential_saved=true" in text
    assert "login_ok=true" in text
    assert requests[0][0] == "/api/oauth/device_authorization"
    assert requests[0][1]["client_id"] == ["17e5f671-d194-4dfb-9706-5516cb48c098"]
    assert requests[1][0] == "/api/oauth/token"
    assert requests[1][1]["device_code"] == ["kimi-device-code"]
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["kimi-code"]["credentials"]["access"] == "kimi-access-token"
    assert payload["kimi-code"]["loginProfile"] == "kimi_code"


def test_oauth_browser_open_uses_windows_fallback_when_webbrowser_fails(monkeypatch) -> None:
    from sim_agent.cli import tui_browser_launch

    calls: list[tuple[str, ...]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.delenv("ASA_BROWSER_OAUTH_OPEN", raising=False)
    monkeypatch.setenv("WSL_INTEROP", "/tmp/wsl-interop")
    monkeypatch.setattr(tui_browser_launch.webbrowser, "open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(tui_browser_launch.subprocess, "run", fake_run)

    opened = tui_browser_launch.open_url_in_browser("https://auth.example.test/start?x=1&y=2", ())

    assert opened is True
    assert calls
    assert "https://auth.example.test/start?x=1&y=2" in calls[0]


def test_oauth_browser_open_prefers_windows_browser_in_wsl(monkeypatch) -> None:
    from sim_agent.cli import tui_browser_launch

    webbrowser_calls: list[str] = []
    subprocess_calls: list[tuple[str, ...]] = []

    def fake_webbrowser_open(url: str, *_args: object, **_kwargs: object) -> bool:
        webbrowser_calls.append(url)
        return True

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.delenv("ASA_BROWSER_OAUTH_OPEN", raising=False)
    monkeypatch.setenv("WSL_INTEROP", "/tmp/wsl-interop")
    monkeypatch.setattr(tui_browser_launch.webbrowser, "open", fake_webbrowser_open)
    monkeypatch.setattr(tui_browser_launch.subprocess, "run", fake_run)

    opened = tui_browser_launch.open_url_in_browser("https://auth.example.test/start", ())

    assert opened is True
    assert webbrowser_calls == []
    assert subprocess_calls


def _write_runtime_config(path: Path) -> Path:
    save_runtime_config(default_runtime_config(), path)
    return path


def _run_module_interactive(lines: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )


def _run_tui(lines: list[str], config_path: Path, *, session_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
