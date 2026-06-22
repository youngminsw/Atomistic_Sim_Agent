from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_workflow_harness_smoke_writes_resumable_state_machine_ledger(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "deep-interview",
        {"request_id": "workflow-smoke", "user_goal": "Clarify runtime requirements"},
        tmp_path,
    )
    ledger = json.loads((tmp_path / result.ledger_ref).read_text(encoding="utf-8"))

    assert result.status == "ready"
    assert result.workflow_id == "deep-interview"
    assert result.current_state == "handoff_ready"
    assert result.resumable is True
    assert result.verification_gate == "ambiguity_gate_clear"
    assert ledger["workflow_id"] == "deep-interview"
    assert ledger["resumable"] is True
    assert [event["state"] for event in ledger["events"]][-1] == "handoff_ready"


def test_tui_workflow_slash_commands_start_harnesses_and_show_ledgers(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    workflow_dir = tmp_path / "workflows"

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input=(
            f"/workflow deep-interview --output-dir {workflow_dir}\n"
            f"/ralplan --output-dir {workflow_dir}\n"
            "/help\n"
            "/exit\n"
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_harness_ready=true" in result.stdout
    assert "workflow=deep-interview" in result.stdout
    assert "workflow=ralplan" in result.stdout
    assert "current_state=handoff_ready" in result.stdout
    assert "current_state=verification_plan_ready" in result.stdout
    assert "/workflow <name>" in result.stdout
    assert "/ralplan" in result.stdout
    assert (workflow_dir / "deep-interview" / "workflow_harness_ledger.json").is_file()
    assert (workflow_dir / "ralplan" / "workflow_harness_ledger.json").is_file()


def test_workflow_harness_unknown_ids_write_only_fixed_unknown_ledger(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("../../escape_workflow", {"request_id": "bad"}, tmp_path)

    assert result.status == "blocked"
    assert result.workflow_id == "unknown"
    assert result.ledger_ref == "unknown/workflow_harness_ledger.json"
    assert (tmp_path / result.ledger_ref).is_file()
    assert not (tmp_path.parent / "escape_workflow").exists()
