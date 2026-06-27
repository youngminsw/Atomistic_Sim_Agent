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
        {
            "request_id": "workflow-smoke",
            "user_goal": "Clarify runtime requirements",
            "evidence": {"question_answer": "surface transcript", "ambiguity_score": 0.2},
        },
        tmp_path,
    )
    ledger = json.loads((tmp_path / result.ledger_ref).read_text(encoding="utf-8"))

    assert result.status == "blocked"
    assert result.workflow_id == "deep-interview"
    assert result.current_state == "blocked"
    assert result.resumable is True
    assert result.verification_gate == "ambiguity_gate_clear"
    assert result.gate_status == "awaiting_response"
    assert ledger["workflow_id"] == "deep-interview"
    assert ledger["resumable"] is True
    assert ledger["gate_status"] == "awaiting_response"
    assert ledger["missing_evidence"] == []
    assert ledger["artifact_refs"] == []
    assert ledger["gate"]["gate_kind"] == "response_schema"
    assert [event["state"] for event in ledger["events"]][-1] == "blocked"
    assert (tmp_path / "deep-interview" / "gates" / "question-q1.json").is_file()
    assert not (tmp_path / "deep-interview" / "handoff.md").exists()


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
            f"/workflow deep-interview --evidence-key question_answer,ambiguity_score --output-dir {workflow_dir}\n"
            f"/ralplan --evidence-key prd_path,test_spec_path --gate-id approval --gate-kind enum "
            f"--allowed-values approve,revise --output-dir {workflow_dir}\n"
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
    assert "current_state=blocked" in result.stdout
    assert "workflow_gate_status=awaiting_response" in result.stdout
    assert "workflow_gate_id=approval" in result.stdout
    assert "workflow_artifact_refs=ralplan/prd.md,ralplan/test-spec.md,ralplan/consensus.json" in result.stdout
    assert "/workflow <name>" in result.stdout
    assert "/ralplan" in result.stdout
    assert (workflow_dir / "deep-interview" / "workflow_harness_ledger.json").is_file()
    assert (workflow_dir / "ralplan" / "workflow_harness_ledger.json").is_file()
    assert (workflow_dir / "ralplan" / "prd.md").is_file()
    assert (workflow_dir / "ralplan" / "test-spec.md").is_file()
    assert (workflow_dir / "ralplan" / "consensus.json").is_file()


def test_workflow_harness_unknown_ids_write_only_fixed_unknown_ledger(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("../../escape_workflow", {"request_id": "bad"}, tmp_path)

    assert result.status == "blocked"
    assert result.workflow_id == "unknown"
    assert result.ledger_ref == "unknown/workflow_harness_ledger.json"
    assert (tmp_path / result.ledger_ref).is_file()
    assert not (tmp_path.parent / "escape_workflow").exists()


def test_workflow_harness_missing_required_evidence_blocks_and_records_gate(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("ralplan", {"request_id": "missing-evidence", "evidence": {}}, tmp_path)
    ledger = json.loads((tmp_path / result.ledger_ref).read_text(encoding="utf-8"))

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("workflow_gate_missing_evidence",)
    assert result.missing_evidence == ("prd_path", "test_spec_path")
    assert ledger["gate_status"] == "blocked"
    assert ledger["missing_evidence"] == ["prd_path", "test_spec_path"]
    assert ledger["events"][-1]["state"] == "blocked"
