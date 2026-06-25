from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_asa_interactive_exposes_agent_team_skill_status_and_logs(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    team_dir = tmp_path / "team"
    result = _run_module_interactive(
        [
            "/agents",
            "/harness",
            "/skills",
            f"/team --output-dir {shlex.quote(str(team_dir))}",
            "/team contract",
            "/status",
            "/log --limit 10",
            "/exit",
        ],
        session_dir=session_dir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent=md_agent" in result.stdout
    assert "agent=qa_agent" in result.stdout
    assert "harness_contract=true" in result.stdout
    assert "call=md_agent->orchestrator,research_agent,qa_agent" in result.stdout
    assert "qa_gate=slurm_job_script:qa_before_submit" in result.stdout
    assert "Agent Workboard" in result.stdout
    assert "role-local harness initialized" in result.stdout
    assert "heartbeat 3600s" in result.stdout
    assert "skill=feature-scale" in result.stdout
    assert "team_session_ready=true" in result.stdout
    assert "team_status=ready" in result.stdout
    assert f"team_ledger_path={team_dir / 'agent_team_session_ledger.json'}" in result.stdout
    assert "team_contract=true" in result.stdout
    assert "session_status=true" in result.stdout
    assert "event=team_session" in result.stdout
    assert (session_dir / "asa_session.json").is_file()
    assert (team_dir / "agent_team_session_ledger.json").is_file()


def test_asa_interactive_exposes_agents_sdk_runtime_surface(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    session_dir = tmp_path / "session"
    result = _run_module_interactive(
        [
            f"/runtime --output-dir {shlex.quote(str(runtime_dir))}",
            "/status",
            "/exit",
        ],
        session_dir=session_dir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime_dry_run=true" in result.stdout
    assert "handoff=md_agent" in result.stdout
    assert "handoff=qa_agent" in result.stdout
    assert "runtime_ledger_path=" in result.stdout
    assert "runtime_ledger=" in result.stdout
    assert (runtime_dir / "agents_sdk_runtime_ledger.json").is_file()
    session = json.loads((session_dir / "asa_session.json").read_text(encoding="utf-8"))
    assert session["runtime_ledger"] == str(runtime_dir / "agents_sdk_runtime_ledger.json")


def _run_module_interactive(
    lines: list[str],
    *,
    session_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    if session_dir is not None:
        env["ASA_SESSION_DIR"] = str(session_dir)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
