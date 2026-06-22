from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

from sim_agent.runtime_config import default_runtime_config, load_runtime_config, save_runtime_config


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_model_profiles_command_lists_codex_style_presets(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(config_path, tmp_path / "session", "/model profiles\n/exit\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_profiles=true" in result.stdout
    assert "model_profile=codex-eco" in result.stdout
    assert "default=openai-codex/gpt-5.5:low" in result.stdout
    assert "agent_model=md_agent:openai-codex/gpt-5.5:minimal" in result.stdout
    assert "model_profile=codex-pro" in result.stdout
    assert "agent_model=qa_agent:openai-codex/gpt-5.5:xhigh" in result.stdout


def test_model_profile_applies_default_and_agent_assignments(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        config_path,
        tmp_path / "session",
        "/model profile codex-pro\n/model status\n/exit\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_profile_saved=codex-pro" in result.stdout
    assert "active_profile=codex-pro" in result.stdout
    assert "profile_customized=false" in result.stdout
    assert "provider=openai-codex model=gpt-5.5 reasoning_effort=xhigh" in result.stdout
    assert "agent=md_agent provider=openai-codex model=gpt-5.5 reasoning_effort=medium override=true" in result.stdout
    assert "agent=qa_agent provider=openai-codex model=gpt-5.5 reasoning_effort=xhigh override=true" in result.stdout
    assert load_runtime_config(config_path).active_profile.name == "codex-pro"
    assert load_runtime_config(config_path).active_profile.customized is False


def test_profile_status_surfaces_agree_after_reload(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    first = _run_tui(config_path, tmp_path / "session-a", "/model profile codex-medium\n/exit\n")
    second = _run_tui(config_path, tmp_path / "session-b", "/model status\n/hud\n/status\n/exit\n")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.count("active_profile=codex-medium") >= 3
    assert "model_profile=codex-medium customized=false" in second.stdout
    assert "model=openai-codex/gpt-5.5/medium/oauth" in second.stdout
    assert "credential_store=" not in second.stdout


def test_manual_endpoint_change_marks_active_profile_customized(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        config_path,
        tmp_path / "session",
        (
            "/model profile codex-pro\n"
            "/model use openai-codex/gpt-5.3-codex-spark --reasoning-effort high\n"
            "/model status\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "active_profile=codex-pro" in result.stdout
    assert "profile_customized=true" in result.stdout
    assert load_runtime_config(config_path).active_profile.name == "codex-pro"
    assert load_runtime_config(config_path).active_profile.customized is True


def test_external_agent_override_divergence_displays_customized_profile(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    first = _run_tui(config_path, tmp_path / "session-a", "/model profile codex-pro\n/exit\n")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["agent_model_overrides"][0]["reasoning_effort"] = "low"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    second = _run_tui(config_path, tmp_path / "session-b", "/model status\n/exit\n")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "active_profile=codex-pro" in second.stdout
    assert "profile_customized=true" in second.stdout


def test_thinking_command_accepts_codex_style_levels(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(config_path, tmp_path / "session", "/model thinking minimal\n/model thinking off\n/exit\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "thinking_level=minimal reasoning_effort=minimal" in result.stdout
    assert "thinking_level=off reasoning_effort=off" in result.stdout


def _write_runtime_config(path: Path) -> Path:
    save_runtime_config(default_runtime_config(), path)
    return path


def _run_tui(config_path: Path, session_dir: Path, commands: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input=commands,
        text=True,
        capture_output=True,
        check=False,
    )
