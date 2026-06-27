from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_audit_runtime_spines_writes_gap_matrix(tmp_path: Path) -> None:
    out_path = tmp_path / "runtime-spines.json"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "ledger.jsonl").write_text(
        json.dumps({"event": "IntegrationBlocker", "task": "full-pytest"}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "audit_runtime_spines.py"),
            "--root",
            str(SOURCE_ROOT),
            "--out",
            str(out_path),
            "--evidence-root",
            str(evidence_root),
        ],
        cwd=SOURCE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime_spine_audit_path=" in result.stdout
    assert payload["status"] == "incomplete"
    assert payload["summary"]["total_spines"] == 8
    assert payload["summary"]["complete_spines"] == 0
    assert payload["summary"]["gap_open"] == 0
    assert payload["summary"]["required_detector_failure_count"] == 0
    assert payload["summary"]["required_detector_failures"] == {}
    assert payload["summary"]["readiness_failure_count"] > 0
    assert "unresolved_blocker_full-pytest" in payload["summary"]["readiness_failure_codes"]
    assert any(
        code.endswith("_missing_red_evidence") or code.startswith("unresolved_blocker_")
        for code in payload["summary"]["readiness_failure_codes"]
    )
    assert payload["spines"]["agent_session"]["detectors"]["agent_session_contract_defined"] is True
    assert payload["spines"]["agent_session"]["detectors"]["mutable_session_history"] is True
    assert payload["spines"]["agent_session"]["readiness_status"] == "incomplete"
    agent_loop_detectors = payload["spines"]["agent_loop"]["detectors"]
    assert "one_shot_choose_tools" not in agent_loop_detectors
    assert agent_loop_detectors["required_model_turn_bridge"] is True
    assert agent_loop_detectors["required_tool_results_appended"] is True
    assert agent_loop_detectors["required_tool_result_continuation_gate"] is True
    assert agent_loop_detectors["required_model_tool_events"] is True


def test_audit_plan_completion_reports_missing_legacy_todo(tmp_path: Path) -> None:
    out_path = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "audit_plan_completion.py"),
            "--root",
            str(SOURCE_ROOT),
            "--todo",
            "1",
            "--out",
            str(out_path),
        ],
        cwd=SOURCE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["todo_id"] == "1"
    assert payload["todo_found"] is False
    assert payload["todo_line"] == ""


def test_audit_scope_fidelity_allows_todo_one_paths(tmp_path: Path) -> None:
    out_path = tmp_path / "scope.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "audit_scope_fidelity.py"),
            "--root",
            str(SOURCE_ROOT),
            "--out",
            str(out_path),
        ],
        cwd=SOURCE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["status"] in {"clean", "scope_review_required"}
    assert "02.Source_code/mss_agent" in payload["forbidden_path_fragments"]
