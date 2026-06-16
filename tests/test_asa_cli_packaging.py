from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tomllib
from io import StringIO
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_pyproject_exports_asa_console_scripts() -> None:
    pyproject = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    dependencies = pyproject["project"]["dependencies"]

    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert scripts["asa"] == "sim_agent.cli.main:main"
    assert scripts["atomistic-sim-agent"] == "sim_agent.cli.main:main"
    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert "sim_agent" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]


def test_asa_module_chat_prepares_main_orchestrator_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "asa-chat-run"
    result = _run_module(
        [
            "chat",
            "--message",
            "Plan Ar etching on amorphous Si with a 3D hole pattern",
            "--output-dir",
            str(output_dir),
            "--source-root",
            str(SOURCE_ROOT),
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "asa_chat_ok=true" in result.stdout
    assert "orchestrator=main" in result.stdout
    assert "agent_run_ledger_path=" in result.stdout
    request = json.loads((output_dir / "validated_request.json").read_text(encoding="utf-8"))
    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    assert request["user_goal"] == "Plan Ar etching on amorphous Si with a 3D hole pattern"
    assert request["scene"]["surface_state"]["phase"] == "amorphous"
    assert request["recipe"]["ion_species"] == "Ar"
    assert ledger["pipeline_stages"][0] == "agent_plan"


def test_asa_module_auth_login_status_redacts_token(tmp_path: Path) -> None:
    store = tmp_path / "credentials.json"
    login = _run_module(
        [
            "auth",
            "login",
            "--provider",
            "oauth_gateway",
            "--access-token",
            "asa-secret-token",
            "--refresh-token",
            "asa-refresh-token",
            "--credential-store",
            str(store),
        ]
    )
    status = _run_module(
        [
            "auth",
            "status",
            "--credential-store",
            str(store),
        ]
    )

    assert login.returncode == 0, login.stdout + login.stderr
    assert status.returncode == 0, status.stdout + status.stderr
    assert "auth_login_ok=true" in login.stdout
    assert "oauth_gateway" in status.stdout
    assert "asa-secret-token" not in status.stdout


def test_asa_module_without_args_opens_interactive_shell() -> None:
    result = _run_module_interactive(["/help", "/exit"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Atomistic Simulation Agent" in result.stdout
    assert "╭" in result.stdout or "+ Atomistic Simulation Agent" in result.stdout
    assert "Agent Workboard" in result.stdout
    assert "asa>" in result.stdout
    assert "/model" in result.stdout
    assert "/run" in result.stdout
    assert "/team" in result.stdout
    assert "/agents" in result.stdout
    assert "/setup" in result.stdout


def test_asa_interactive_slash_opens_command_palette() -> None:
    result = _run_module_interactive(["/", "/exit"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Slash Command Palette" in result.stdout
    assert "/runtime" in result.stdout
    assert "/setup" in result.stdout
    assert "simulation skills" in result.stdout
    assert "md" in result.stdout


def test_asa_interactive_login_selector_and_api_key_mode_redacts_secret(tmp_path: Path) -> None:
    store = tmp_path / "credentials.json"
    result = _run_module_interactive(
        [
            "/login",
            "/login api-key "
            f"--provider openai --api-key cli-secret-token --credential-store {shlex.quote(str(store))}",
            "/model status",
            "/exit",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Login Options" in result.stdout
    assert "/login oauth --provider" in result.stdout
    assert "/login api-key --provider" in result.stdout
    assert "login_ok=true" in result.stdout
    assert "provider=openai logged_in=True" in result.stdout
    assert "cli-secret-token" not in result.stdout


def test_asa_bare_login_interactive_selector_persists_api_key(tmp_path: Path, monkeypatch) -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    from sim_agent.cli.tui_login import LoginMode, handle_login
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV

    class FakeSelector:
        def choose_mode(self) -> LoginMode:
            return "api_key"

        def prompt_provider(self, default: str) -> str:
            assert default == "openclaw"
            return "openai"

        def prompt_token(self, mode: LoginMode) -> str:
            assert mode == "api_key"
            return "interactive-secret-token"

    store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(store))
    output = StringIO()

    handle_login((), initial_state(tmp_path), output, FakeSelector())

    assert "login_ok=true" in output.getvalue()
    assert "provider=openai auth_mode=api_key" in output.getvalue()
    assert "interactive-secret-token" not in output.getvalue()
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["openai"]["provider"] == "openai"


def test_asa_live_slash_completion_catalog_exposes_commands_and_skills() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    from sim_agent.cli.tui_prompt import slash_completion_rows

    rows = slash_completion_rows("/")
    command_rows = {row.value: row for row in rows if row.kind == "command"}
    skill_rows = {row.value: row for row in rows if row.kind == "skill"}

    assert "/model" in command_rows
    assert "/login" in command_rows
    assert "/harness" in command_rows
    assert "/runtime" in command_rows
    assert "/skills" in command_rows
    assert "gateway/model" in command_rows["/model"].meta
    assert "md" in skill_rows
    assert "LAMMPS" in skill_rows["md"].meta


def test_asa_interactive_model_login_status_redacts_token(tmp_path: Path) -> None:
    store = tmp_path / "credentials.json"
    result = _run_module_interactive(
        [
            "/model login "
            f"--provider oauth_gateway --access-token asa-secret-token "
            f"--refresh-token asa-refresh-token --credential-store {shlex.quote(str(store))}",
            "/model status",
            "/exit",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_login_ok=true" in result.stdout
    assert "provider=oauth_gateway logged_in=True" in result.stdout
    assert "asa-secret-token" not in result.stdout


def test_asa_interactive_run_prepares_orchestrator_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "asa-tui-run"
    result = _run_module_interactive(
        [
            "/run "
            f"--output-dir {shlex.quote(str(output_dir))} "
            f"--source-root {shlex.quote(str(SOURCE_ROOT))} "
            "Plan Ar etching on amorphous Si with a 3D hole pattern",
            "/exit",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_id=plan-cli_ar_si_amorphous_hole" in result.stdout
    assert (output_dir / "agent_run_ledger.json").is_file()


def test_asa_interactive_accepts_session_dir_cli_option(tmp_path: Path) -> None:
    session_dir = tmp_path / "explicit-session"
    result = _run_module_interactive(
        [
            "/status",
            "/exit",
        ],
        extra_args=["--session-dir", str(session_dir)],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"session_dir={session_dir}" in result.stdout
    assert (session_dir / "asa_session.json").is_file()


def _run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_module_interactive(
    lines: list[str],
    *,
    session_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    if session_dir is not None:
        env["ASA_SESSION_DIR"] = str(session_dir)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent", *(extra_args or [])],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
