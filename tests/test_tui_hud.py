from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_asa_tui_hud_warns_when_model_credentials_are_missing(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = _run_interactive(["/hud", "/model status", "/exit"], env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "hud=true" in result.stdout
    assert "model_connected=False" in result.stdout
    assert "Model is not connected" in result.stdout
    assert "/login" in result.stdout
    assert "/model set" in result.stdout


def test_asa_tui_hud_reports_connected_provider_after_login(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = _run_interactive(
        [
            "/login api-key --provider openai --api-key test-token",
            "/model set --provider openai --model gpt-5.5 --auth-mode api_key",
            "/hud",
            "/exit",
        ],
        env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_connected=True" in result.stdout
    assert "provider=openai model=gpt-5.5 auth_mode=api_key" in result.stdout
    assert "test-token" not in result.stdout


def test_malformed_stored_credentials_are_not_reported_connected(monkeypatch, tmp_path: Path) -> None:
    credential_store = tmp_path / "credentials.json"
    credential_store.write_text(
        json.dumps(
            {
                "openai": {
                    "provider": "openai",
                    "credentials": {"authMode": "oauth", "expires": 4102444800000},
                    "updatedAtMs": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE", str(credential_store))

    from sim_agent.schemas._parse import as_mapping, as_sequence
    from sim_agent.ui.model_auth import model_auth_status_payload
    from sim_agent.ui.model_connection import model_connection_status

    auth_payload = model_auth_status_payload()
    providers = as_sequence(auth_payload["providers"], "providers")
    provider_status = as_mapping(providers[0], "providers[0]")
    connection = model_connection_status("openai", "gpt-5.5", "oauth", "OPENAI_API_KEY")

    assert provider_status["logged_in"] is False
    assert auth_payload["connected_provider_count"] == 0
    assert connection.connected is False
    assert "missing a usable access token" in connection.friendly_message


def test_asa_slash_catalog_exposes_hud_command() -> None:
    env = _env(Path(tempfile.gettempdir()))
    result = _run_interactive(["/", "/exit"], env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "/hud" in result.stdout
    assert "provider/model/auth/session HUD" in result.stdout


def _run_interactive(lines: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE"] = str(tmp_path / "credentials.json")
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    env.pop("OPENCLAW_OAUTH_TOKEN", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("MODEL_GATEWAY_TOKEN", None)
    return env
