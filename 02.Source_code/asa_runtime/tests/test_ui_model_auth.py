from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_provider_credentials_default_to_asa_home_and_avoid_gateway_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.ui.model_auth import login_model_provider, model_auth_status_payload

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE", raising=False)
    monkeypatch.delenv("ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE", raising=False)

    login = login_model_provider(
        {
            "provider": "openai-codex",
            "access_token": "provider-token",
            "refresh_token": "provider-refresh",
            "auth_mode": "oauth",
        }
    )
    status = model_auth_status_payload()

    expected = home / ".asa" / "provider-credentials.json"
    assert login["provider_credential_store"] == str(expected)
    assert "credential_store" not in login
    assert expected.is_file()
    assert not (home / ".asa" / "model-gateway-credentials.json").exists()
    assert status["provider_credential_store"] == str(expected)
    assert "model-gateway-credentials" not in json.dumps(status)


def test_legacy_gateway_credential_store_is_migrated_to_provider_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.ui.model_auth import access_token_for_provider, model_auth_status_payload

    home = tmp_path / "home"
    legacy = home / ".atomistic-sim-agent" / "model-gateway-credentials.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "provider": "openai-codex",
                    "credentials": {
                        "access": "legacy-access",
                        "refresh": "legacy-refresh",
                        "expires": 4_102_444_800_000,
                        "authMode": "oauth",
                    },
                    "updatedAtMs": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE", raising=False)
    monkeypatch.delenv("ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE", raising=False)

    token = access_token_for_provider("openai-codex")
    status = model_auth_status_payload()

    migrated = home / ".asa" / "provider-credentials.json"
    assert token == "legacy-access"
    assert migrated.is_file()
    assert status["provider_credential_store"] == str(migrated)
    assert "model-gateway-credentials" not in json.dumps(status)


def test_legacy_gateway_credential_env_alias_still_reads_custom_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.ui.model_auth import access_token_for_provider

    store = tmp_path / "legacy-env-credentials.json"
    store.write_text(
        json.dumps(
            {
                "anthropic": {
                    "provider": "anthropic",
                    "credentials": {"access": "legacy-env-access", "authMode": "oauth"},
                    "updatedAtMs": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE", raising=False)
    monkeypatch.setenv("ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE", str(store))

    assert access_token_for_provider("anthropic") == "legacy-env-access"


def test_controller_model_auth_login_status_and_gateway_smoke(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    credential_store = tmp_path / "credentials.json"
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE", str(credential_store))
    monkeypatch.setenv("ATOMISTIC_MODEL_GATEWAY_SMOKE_DIR", str(tmp_path / "smoke"))
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
    gateway_host = gateway.server_name
    gateway_port = gateway.server_port
    gateway_thread = Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host = server.server_name
    port = server.server_port
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        login_body, login_code = _post_json(
            f"http://{host}:{port}/api/model/auth/login",
            {
                "provider": "local_gateway",
                "access_token": "controller-secret-token",
                "refresh_token": "controller-refresh-token",
                "expires_in_s": 3600,
            },
        )
        status_body = as_mapping(
            json.loads(urlopen(f"http://{host}:{port}/api/model/auth/status", timeout=5).read().decode("utf-8")),
            "model_auth_status",
        )
        smoke_body, smoke_code = _post_json(
            f"http://{host}:{port}/api/model/gateway/smoke",
            {
                "model_provider": {
                    "provider": "local_gateway",
                    "model": "gpt-5.5",
                    "reasoning_effort": "high",
                    "base_url": f"http://{gateway_host}:{gateway_port}/v1",
                    "auth_mode": "gateway",
                    "api_key_env": "RUNTIME_GATEWAY_TOKEN",
                },
                "request": {
                    "request_id": "controller-smoke",
                    "ion_species": "Ar",
                    "target_material": "Si",
                },
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        gateway.shutdown()
        gateway.server_close()
        gateway_thread.join(timeout=5)

    assert login_code == 200
    assert login_body["ok"] is True
    assert login_body["provider"] == "local_gateway"
    assert "credential_store" not in login_body
    assert "controller-secret-token" not in json.dumps(login_body)
    provider_status = _first_provider_status(status_body)
    assert provider_status["provider"] == "local_gateway"
    assert provider_status["logged_in"] is True
    assert "controller-secret-token" not in json.dumps(status_body)
    assert smoke_code == 200
    assert smoke_body["ok"] is True
    assert smoke_body["gateway_request_id"] == "controller-gw-1"


class _GatewayHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        match self.path:
            case "/healthz":
                self._write_json({"ok": True})
            case "/v1/models":
                self._write_json({"object": "list", "data": [{"id": "gpt-5.5"}]})
            case _:
                self._write_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write_json({"error": "not_found"}, 404)
            return
        if self.headers.get("Authorization") != "Bearer controller-secret-token":
            self._write_json({"error": {"code": "missing_gateway_credentials"}}, 401)
            return
        self._write_json(
            {
                "id": "resp_controller_gw_1",
                "gateway_request_id": "controller-gw-1",
                "output_text": "controller_gateway_ready",
            }
        )

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, payload: JsonMap, status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _post_json(url: str, payload: JsonMap) -> tuple[JsonMap, int]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-ASA-CSRF-Token": "test-token"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return as_mapping(json.loads(exc.read().decode("utf-8")), "response"), exc.code
    return as_mapping(json.loads(response.read().decode("utf-8")), "response"), response.status


def _first_provider_status(payload: JsonMap) -> JsonMap:
    providers = payload.get("providers")
    if not isinstance(providers, list) or len(providers) == 0:
        raise AssertionError("providers_missing")
    return as_mapping(providers[0], "providers[0]")
