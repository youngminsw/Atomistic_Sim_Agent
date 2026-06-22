from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
VALID_RUN = FIXTURE_ROOT / "md_runs" / "small_valid"
MATERIAL_ROOT = FIXTURE_ROOT / "materials"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_postprocess_lammps_execution_result_parses_and_verifies_completed_run(
    tmp_path: Path,
) -> None:
    from sim_agent.md import postprocess_lammps_execution_result

    # Given
    events_path = tmp_path / "md_events.jsonl"

    # When
    report = postprocess_lammps_execution_result(
        _execution_result_payload("lammps_completed", VALID_RUN),
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        events_out=events_path,
    )

    # Then
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert report.ok is True
    assert report.payload["postprocess_status"] == "md_postprocess_complete"
    assert report.payload["event_count"] == 2
    assert report.payload["layer_removed_count"] == 1
    assert report.payload["verification_status"] == "verified"
    assert "md_events_parsed" in report.payload["evidence"]
    assert rows[0]["event_id"] == "evt-0001"


def test_postprocess_lammps_execution_result_rejects_unfinished_execution(
    tmp_path: Path,
) -> None:
    from sim_agent.md import postprocess_lammps_execution_result

    # Given
    events_path = tmp_path / "md_events.jsonl"

    # When
    report = postprocess_lammps_execution_result(
        _execution_result_payload("lammps_failed", VALID_RUN, return_code=2),
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        events_out=events_path,
    )

    # Then
    assert report.ok is False
    assert report.payload["postprocess_status"] == "lammps_execution_not_complete"
    assert report.payload["errors"] == ["lammps_execution_status=lammps_failed"]
    assert not events_path.exists()


def test_postprocess_lammps_execution_result_rejects_event_count_mismatch(
    tmp_path: Path,
) -> None:
    from sim_agent.md import postprocess_lammps_execution_result

    events_path = tmp_path / "md_events.jsonl"

    report = postprocess_lammps_execution_result(
        _execution_result_payload("lammps_completed", VALID_RUN) | {"expected_incident_count": 3},
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        events_out=events_path,
    )

    assert report.ok is False
    assert report.payload["postprocess_status"] == "md_verification_failed"
    assert report.payload["expected_incident_count"] == 3
    assert "event_count_mismatch:expected=3:actual=2" in report.payload["errors"]


def test_postprocess_lammps_execution_cli_writes_report(tmp_path: Path) -> None:
    # Given
    execution_result_path = tmp_path / "lammps_execution_result.json"
    events_path = tmp_path / "md_events.jsonl"
    report_path = tmp_path / "md_postprocess_report.json"
    execution_result_path.write_text(
        json.dumps(_execution_result_payload("lammps_completed", VALID_RUN)),
        encoding="utf-8",
    )

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "postprocess_lammps_execution.py"),
            "--execution-result",
            str(execution_result_path),
            "--material",
            "Si",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--events-out",
            str(events_path),
            "--report-out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(report_path.read_text(encoding="utf-8")), "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_postprocess_ok=true" in result.stdout
    assert "postprocess_status=md_postprocess_complete" in result.stdout
    assert payload["event_count"] == 2
    assert payload["verification_status"] == "verified"


def _execution_result_payload(
    status: str,
    run_dir: Path,
    return_code: int = 0,
) -> JsonMap:
    return {
        "execution_result_id": "run-1-lammps-execution-result",
        "execution_plan_id": "run-1-lammps-execution-plan",
        "run_id": "run-1",
        "execution_status": status,
        "preflight_ok": True,
        "execute_requested": True,
        "working_directory": str(run_dir),
        "command": ["lmp", "-in", "in.atomistic_campaign"],
        "command_line": f"cd {run_dir} && lmp -in in.atomistic_campaign",
        "required_inputs": ["in.atomistic_campaign"],
        "expected_outputs": ["log.lammps"],
        "missing_expected_outputs": [],
        "return_code": return_code,
        "stdout": "",
        "stderr": "",
    }
