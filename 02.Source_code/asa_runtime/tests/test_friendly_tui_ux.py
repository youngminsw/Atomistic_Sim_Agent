from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_tui_guide_gives_plain_language_korean_onboarding_and_model_visibility(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/guide\n/model status\n/help\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "friendly_guide=true" in result.stdout
    assert "초보자 안내" in result.stdout
    assert "model_visible=" in result.stdout
    assert "next_step=/model status" in result.stdout
    assert "next_step=/workflow deep-interview" in result.stdout
    assert "plain_goal_hint=그냥 하고 싶은 시뮬레이션을 문장으로 입력해도 됩니다" in result.stdout
    assert "/guide" in result.stdout
    assert "provider=" in result.stdout


def test_tui_beginner_start_and_help_are_visible_without_cli_flags(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/help\n/start\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "처음이면 /guide 또는 /start" in result.stdout
    assert "초보자 빠른 시작" in result.stdout
    assert "beginner_help=true" in result.stdout
    assert "beginner_first_step=/guide" in result.stdout
    assert "beginner_runtime_test=/runtime tools" in result.stdout
    assert "명령어를 외우지 않아도 됩니다" in result.stdout
    assert "friendly_guide=true" in result.stdout
    assert "next_step=/memory" in result.stdout


def test_tui_memory_shows_graphdb_brain_query_plan(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/memory\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "graph_memory=true" in result.stdout
    assert "graph_memory_status=query_plan_ready" in result.stdout
    assert "research_write_owner=research_graphdb_agent" in result.stdout
    assert "agent_brain=md_agent:query_planned:evidence=0" in result.stdout
    assert "agent_brain=qa_agent:query_planned:evidence=0" in result.stdout
    assert "live_check_hint=/memory live" in result.stdout


def test_tui_memory_live_fails_closed_without_neo4j_credentials(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/memory live\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "graph_memory=true" in result.stdout
    assert "graph_memory_status=blocked" in result.stdout
    assert "graph_memory_blocker=missing_env:NEO4J_USERNAME" in result.stdout


def test_tui_memory_live_fails_closed_on_unreachable_neo4j(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    env["NEO4J_URI"] = "bolt://127.0.0.1:1"
    env["NEO4J_USERNAME"] = "neo4j"
    env["NEO4J_PASSWORD"] = "not-a-real-password"
    env["NEO4J_DATABASE"] = "atomistic_sim_agent_knowledge"

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/memory live\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "graph_memory=true" in result.stdout
    assert "graph_memory_status=blocked" in result.stdout
    assert "graph_memory_blocker=ServiceUnavailable" in result.stdout


def test_tui_model_set_preserves_reasoning_effort_in_runtime(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input=(
            "/model set --provider local_gateway --model gpt-5.3 "
            "--reasoning-effort low --base-url http://local-gateway.test/v1 --auth-mode none\n"
            "/model status\n"
            "/runtime --tool-gateway\n"
            "/status\n"
            "/exit\n"
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    session = json.loads((tmp_path / "session" / "asa_session.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "reasoning_effort=low" in result.stdout
    assert "runtime_error=high_stakes_model_requires_high_reasoning" in result.stdout
    assert session["model"]["reasoning_effort"] == "low"
    assert "model=local_gateway/gpt-5.3/low/none" in result.stdout
