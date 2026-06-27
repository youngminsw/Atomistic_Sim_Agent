from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
CLI_SUBPROCESS_TIMEOUT_S = 30
CREDENTIAL_STORE_ENV = "ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE"


def test_pyproject_exports_asa_console_scripts() -> None:
    pyproject = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    dependencies = pyproject["project"]["dependencies"]

    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert scripts["asa"] == "sim_agent.cli.main:main"
    assert scripts["atomistic-sim-agent"] == "sim_agent.cli.main:main"
    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert any(dependency.startswith("openai-agents") for dependency in dependencies)
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
            "openai-codex",
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
    assert "openai-codex" in status.stdout
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


def test_asa_interactive_can_invoke_markdown_skill_by_slash_name(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    result = _run_module_interactive(
        [
            "/md prepare a tiny LAMMPS verification plan",
            "/research collect provenance for Ar Si etch",
            "/exit",
        ],
        session_dir=session_dir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "markdown_skill_invoked=true" in result.stdout
    assert "skill_name=md" in result.stdout
    assert "skill_agent=md_agent" in result.stdout
    assert "skill_file=" in result.stdout
    assert "skill_name=research" in result.stdout
    assert "skill_agent=research_agent" in result.stdout
    md_messages = _jsonl(tmp_path / "session" / "agent_sessions" / "md_agent" / "messages.jsonl")
    md_context_index = _message_index(md_messages, "system", "ASA_SKILL_CONTEXT_V1")
    md_user_index = _message_index(md_messages, "user", "prepare a tiny LAMMPS verification plan")
    assert md_context_index < md_user_index
    assert "Skill: md" in md_messages[md_context_index]["content"]
    assert "# MD Skill" in md_messages[md_context_index]["content"]
    research_messages = _jsonl(tmp_path / "session" / "agent_sessions" / "research_agent" / "messages.jsonl")
    research_context_index = _message_index(research_messages, "system", "ASA_SKILL_CONTEXT_V1")
    assert "Skill: research" in research_messages[research_context_index]["content"]
    md_messages = (session_dir / "agent_sessions" / "md_agent" / "messages.jsonl").read_text(encoding="utf-8")
    research_messages = (session_dir / "agent_sessions" / "research_agent" / "messages.jsonl").read_text(
        encoding="utf-8"
    )
    assert "ASA_SKILL_CONTEXT_V1" in md_messages
    assert "Skill: md" in md_messages
    assert "system prompt layer" in md_messages
    assert "ASA_SKILL_CONTEXT_V1" in research_messages
    assert "Skill: research" in research_messages


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
    assert "login_company=OpenAI" in result.stdout
    assert "login_company=Google" in result.stdout
    assert "ChatGPT Plus/Pro" in result.stdout
    assert "Google Cloud Code Assist" in result.stdout
    assert "GitHub Copilot" in result.stdout
    assert "Cursor" in result.stdout
    assert "xAI" in result.stdout
    assert "OpenClaw" not in result.stdout
    assert "/login oauth --provider" in result.stdout
    assert "/login api-key --provider" in result.stdout
    assert "Provider [openclaw]" not in result.stdout
    assert "login_ok=true" in result.stdout
    assert "provider=openai logged_in=True" in result.stdout
    assert "cli-secret-token" not in result.stdout


def test_asa_live_slash_completion_catalog_exposes_commands_and_skills(tmp_path: Path, monkeypatch) -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    from sim_agent.cli.tui_prompt import slash_completion_rows

    skill_root = _write_markdown_skill(tmp_path / "skills", "dynamic-skill", "/dynamic-skill", "qa_agent")
    monkeypatch.setenv("ASA_SKILL_ROOTS", str(skill_root))

    rows = slash_completion_rows("/")
    command_rows = {row.value: row for row in rows if row.kind == "command"}
    skill_rows = {row.value: row for row in rows if row.kind == "skill"}

    assert "/model" in command_rows
    assert "/login" in command_rows
    assert "/harness" in command_rows
    assert "/runtime" in command_rows
    assert "/wizard" in command_rows
    assert "/memory" in command_rows
    assert "/skills" in command_rows
    assert "gateway/model" in command_rows["/model"].meta
    assert "/md" in skill_rows
    assert skill_rows["/md"].insert_text == "/md"
    assert "LAMMPS" in skill_rows["/md"].meta
    assert "/dynamic-skill" in skill_rows
    assert skill_rows["/dynamic-skill"].insert_text == "/dynamic-skill"


def test_asa_interactive_slash_skill_without_message_is_blocked() -> None:
    result = _run_module_interactive(["/md", "/exit"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skill_blocked=/md blocker=missing_message" in result.stdout
    assert "markdown_skill_invoked=true" not in result.stdout


def test_asa_interactive_model_login_status_redacts_token(tmp_path: Path) -> None:
    store = tmp_path / "credentials.json"
    result = _run_module_interactive(
        [
            "/model login "
            f"--provider openai-codex --access-token asa-secret-token "
            f"--refresh-token asa-refresh-token --credential-store {shlex.quote(str(store))}",
            "/model status",
            "/exit",
        ]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_login_ok=true" in result.stdout
    assert "provider=openai-codex logged_in=True" in result.stdout
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
    assert "session_dir=" in result.stdout
    assert (session_dir / "asa_session.json").is_file()
    session = json.loads((session_dir / "asa_session.json").read_text(encoding="utf-8"))
    assert session["session_dir"] == str(session_dir)


def _run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    with tempfile.TemporaryDirectory(prefix="asa-test-credentials-") as credentials_dir:
        env[CREDENTIAL_STORE_ENV] = str(Path(credentials_dir) / "provider-credentials.json")
        return subprocess.run(
            [sys.executable, "-m", "sim_agent", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=CLI_SUBPROCESS_TIMEOUT_S,
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
    with tempfile.TemporaryDirectory(prefix="asa-test-credentials-") as credentials_dir:
        env[CREDENTIAL_STORE_ENV] = str(Path(credentials_dir) / "provider-credentials.json")
        return subprocess.run(
            [sys.executable, "-m", "sim_agent", *(extra_args or [])],
            cwd=PROJECT_ROOT,
            env=env,
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=CLI_SUBPROCESS_TIMEOUT_S,
        )

def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _message_index(messages: list[dict[str, object]], role: str, content: str) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == role and content in str(message.get("content", "")):
            return index
    raise AssertionError(f"missing {role} message containing {content}")


def _write_markdown_skill(root: Path, name: str, command: str, agent_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                f"command: {command}",
                f"agent_id: {agent_id}",
                f"summary: {name} summary",
                "---",
                f"# {name}",
                "",
                f"{name} body",
                "",
            )
        ),
        encoding="utf-8",
    )
    return root
