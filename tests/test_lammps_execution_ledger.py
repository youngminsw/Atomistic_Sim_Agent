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


def test_write_lammps_execution_ledger_preserves_md_artifacts(tmp_path: Path) -> None:
    from sim_agent.md import write_lammps_execution_ledger

    # Given
    events_path = tmp_path / "md_events.jsonl"
    events_path.write_text(
        '{"event_id":"evt-0001","ion":"Ar","material_id":"Si"}\n'
        '{"event_id":"evt-0002","ion":"Ar","material_id":"Si"}\n',
        encoding="utf-8",
    )

    # When
    bundle = write_lammps_execution_ledger(
        output_dir=tmp_path / "ledger",
        run_id="run-1",
        worker_capability_payload=_worker_capability_payload(),
        execution_result_payload=_execution_result_payload("lammps_completed", tmp_path),
        postprocess_report_payload=_postprocess_report_payload(),
        events_path=events_path,
    )

    # Then
    payload = as_mapping(json.loads(bundle.ledger_path.read_text(encoding="utf-8")), "ledger")
    artifacts = as_mapping(payload["artifacts"], "artifacts")
    assert bundle.artifact_count == 5
    assert payload["run_id"] == "run-1"
    assert payload["run_status"] == "complete"
    assert payload["worker_capability_gate_status"] == "worker_capability_ready"
    assert payload["execution_status"] == "lammps_completed"
    assert payload["postprocess_status"] == "md_postprocess_complete"
    assert artifacts["md_events"] == "md_events.jsonl"
    assert (tmp_path / "ledger" / "worker_capability.json").exists()
    assert (tmp_path / "ledger" / "lammps_execution_result.json").exists()
    assert (tmp_path / "ledger" / "md_postprocess_report.json").exists()
    assert (tmp_path / "ledger" / "md_events.jsonl").exists()


def test_postprocess_lammps_execution_cli_writes_ledger_dir(tmp_path: Path) -> None:
    # Given
    worker_path = tmp_path / "worker_capability.json"
    execution_result_path = tmp_path / "lammps_execution_result.json"
    events_path = tmp_path / "md_events.jsonl"
    report_path = tmp_path / "md_postprocess_report.json"
    ledger_dir = tmp_path / "ledger"
    worker_path.write_text(json.dumps(_worker_capability_payload()), encoding="utf-8")
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
            "--worker-capability",
            str(worker_path),
            "--ledger-dir",
            str(ledger_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    ledger = as_mapping(
        json.loads((ledger_dir / "ledger.json").read_text(encoding="utf-8")),
        "ledger",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lammps_execution_ledger_path=" in result.stdout
    assert ledger["run_status"] == "complete"
    assert ledger["verification_status"] == "verified"
    assert (ledger_dir / "md_events.jsonl").exists()


def _worker_capability_payload() -> JsonMap:
    return {
        "ok": True,
        "gate_status": "worker_capability_ready",
        "evidence": ["worker_capability_ready", "lammps_required_packages_present"],
        "blockers": [],
    }


def _execution_result_payload(status: str, run_dir: Path) -> JsonMap:
    return {
        "execution_result_id": "run-1-lammps-execution-result",
        "execution_plan_id": "run-1-lammps-execution-plan",
        "run_id": "run-1",
        "execution_status": status,
        "preflight_ok": True,
        "worker_capability_gate_status": "worker_capability_ready",
        "preflight_evidence": ["worker_capability_ready"],
        "execute_requested": True,
        "working_directory": str(run_dir),
        "command": ["lmp", "-in", "in.atomistic_campaign"],
        "command_line": f"cd {run_dir} && lmp -in in.atomistic_campaign",
        "required_inputs": ["in.atomistic_campaign"],
        "expected_outputs": ["log.lammps"],
        "missing_expected_outputs": [],
        "return_code": 0,
        "stdout": "",
        "stderr": "",
    }


def _postprocess_report_payload() -> JsonMap:
    return {
        "postprocess_result_id": "run-1-md-postprocess",
        "execution_result_id": "run-1-lammps-execution-result",
        "run_id": "run-1",
        "postprocess_status": "md_postprocess_complete",
        "ok": True,
        "event_count": 2,
        "layer_removed_count": 1,
        "total_deposited_energy_eV": 93.0,
        "verification_status": "verified",
        "evidence": ["md_events_parsed", "lammps_completed"],
        "errors": [],
    }
