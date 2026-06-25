from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_project_layout_creates_project_local_asa_dirs_without_overwriting_templates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.project_layout import ensure_project_state_layout

    project = _project(tmp_path)
    existing_agents = project / ".asa" / "AGENTS.md"
    existing_agents.parent.mkdir(parents=True)
    existing_agents.write_text("custom agents\n", encoding="utf-8")
    monkeypatch.chdir(project)

    layout = ensure_project_state_layout()

    assert layout.project_root == project
    assert layout.asa_root == project / ".asa"
    assert layout.session_root.is_dir()
    assert layout.evidence_root.is_dir()
    assert layout.skills_root.is_dir()
    assert existing_agents.read_text(encoding="utf-8") == "custom agents\n"
    assert (project / ".asa" / "SKILLS.md").is_file()


def test_initial_state_defaults_to_project_local_asa_and_honors_session_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.cli.tui_state import initial_state

    project = _project(tmp_path)
    runtime_config = tmp_path / "missing-runtime-config.json"
    monkeypatch.chdir(project)
    monkeypatch.delenv("ASA_SESSION_DIR", raising=False)
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG", str(runtime_config))

    state = initial_state()

    assert state.session_dir.parent == project / ".asa" / "sessions"
    assert (project / ".asa" / "global_session_index.jsonl").is_file()

    explicit = tmp_path / "explicit-session"
    env_session = tmp_path / "env-session"
    monkeypatch.setenv("ASA_SESSION_DIR", str(env_session))
    assert initial_state().session_dir == env_session
    assert initial_state(explicit).session_dir == explicit


def test_default_runtime_config_uses_project_root_but_config_file_stays_in_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.runtime_config import default_runtime_config, runtime_config_path

    project = _project(tmp_path)
    home = tmp_path / "home"
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG", raising=False)

    config = default_runtime_config()

    assert runtime_config_path() == home / ".asa" / "runtime-config.json"
    assert config.workspace_root == str(project)
    assert config.evidence_root == str(project / ".asa" / "evidence")


def test_python_module_default_session_creates_project_local_asa_state(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(tmp_path / "missing-runtime-config.json")
    env.pop("ASA_SESSION_DIR", None)

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=project,
        env=env,
        input="/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    sessions_root = project / ".asa" / "sessions"
    sessions = tuple(path for path in sessions_root.iterdir() if path.is_dir())
    assert len(sessions) == 1
    assert (sessions[0] / "asa_session.json").is_file()
    assert (project / ".asa" / "global_session_index.jsonl").is_file()
    index = _jsonl(project / ".asa" / "global_session_index.jsonl")
    assert index[0]["session_dir"].startswith(str(sessions_root))


def test_project_layout_blocks_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.project_layout import ProjectLayoutError, ensure_project_state_layout

    project = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".asa").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(project)

    try:
        ensure_project_state_layout()
    except ProjectLayoutError as exc:
        assert "project_state_path_outside_root" in str(exc)
    else:  # pragma: no cover - regression guard.
        raise AssertionError("symlinked .asa escape was accepted")


def test_missing_asa_project_root_env_is_ignored(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.project_layout import ensure_project_state_layout

    project = _project(tmp_path)
    missing = tmp_path / "missing" / "root"
    monkeypatch.chdir(project)
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(missing))

    layout = ensure_project_state_layout()

    assert layout.project_root == project
    assert not missing.exists()


def test_global_session_index_ignores_outside_entries(tmp_path: Path) -> None:
    from sim_agent.agent_runtime.global_session_store import latest_index_entry, lookup_index

    project = _project(tmp_path)
    asa_root = project / ".asa"
    sessions_root = asa_root / "sessions"
    sessions_root.mkdir(parents=True)
    inside = sessions_root / "asa-inside"
    outside = tmp_path / "outside-session"
    index = asa_root / "global_session_index.jsonl"
    index.write_text(
        "\n".join(
            (
                json.dumps({"session_id": "outside", "session_dir": str(outside)}),
                json.dumps({"session_id": "inside", "session_dir": str(inside)}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert lookup_index(asa_root, "outside") is None
    assert lookup_index(asa_root, "inside") == inside
    assert latest_index_entry(asa_root) == inside


def test_project_guidance_reads_compatibility_files_and_ignores_non_utf8(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from sim_agent.agent_runtime import load_agent_registry
    from sim_agent.agent_runtime.live_agent_context import live_turn_project_guidance
    from sim_agent.cli.tui_state import initial_state

    project = _project(tmp_path)
    (project / ".asa").mkdir()
    (project / ".asa" / "AGENTS.md").write_bytes(b"\xff\xfeinvalid")
    (project / ".asa" / "SKILLS.md").write_text("# Project skills\nUse /project-plan.\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG", str(tmp_path / "missing-runtime-config.json"))

    state = initial_state()
    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    guidance = live_turn_project_guidance(handle)

    assert "[AGENTS.md]" in guidance
    assert "# Project agents" in guidance
    assert "[.asa/SKILLS.md]" in guidance
    assert "Use /project-plan." in guidance
    assert "invalid" not in guidance


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = \"sample\"\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("# Project agents\n", encoding="utf-8")
    return project


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
