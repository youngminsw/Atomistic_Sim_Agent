from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
SUCCESS_LOG = FIXTURE_ROOT / "md_logs" / "success_lammps.log"
FAILED_LOG = FIXTURE_ROOT / "md_logs" / "failed_lammps.log"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_verified_lammps_run_accepts_success_log_and_event_dataset() -> None:
    from sim_agent.md import MDRunStatus, verify_md_run

    report = verify_md_run(
        log_path=SUCCESS_LOG,
        events_path=EVENTS,
        expected_events=2,
        required_ion="Ar",
        required_material="Si",
    )

    assert report.ok is True
    assert report.status == MDRunStatus.VERIFIED
    assert report.dataset is not None
    assert report.dataset.event_count == 2
    assert report.dataset.total_deposited_energy_eV == pytest.approx(93.0)
    assert report.dataset.total_removed_depth_nm == pytest.approx(0.035)
    assert report.dataset.reflected_count == 1
    assert report.dataset.sputtered_count == 1
    assert "lammps_completed" in report.evidence
    assert report.errors == ()


def test_failed_lammps_log_blocks_downstream_even_when_events_exist() -> None:
    from sim_agent.md import MDRunStatus, verify_md_run

    report = verify_md_run(log_path=FAILED_LOG, events_path=EVENTS, expected_events=2)

    assert report.ok is False
    assert report.status == MDRunStatus.FAILED
    assert report.dataset is None
    assert "lammps_lost_atoms" in report.errors


def test_physically_unsane_event_dataset_is_rejected(tmp_path: Path) -> None:
    from sim_agent.md import MDRunStatus, verify_md_run

    bad_events = tmp_path / "bad_events.jsonl"
    bad_events.write_text(
        (
            '{"event_id":"evt-bad","ion":"Ar","material_id":"Si","energy_eV":10.0,'
            '"polar_deg":30.0,"azimuth_deg":0.0,'
            '"surface_state":{"amorphous_index":0.0,"roughness_rms_nm":0.1,"removed_depth_nm":0.0},'
            '"outcome":{"event_type":"sputter","yield_atoms_per_ion":1.0,"reflected":false,'
            '"deposited_energy_eV":25.0,"removed_depth_nm":0.01}}\n'
        ),
        encoding="utf-8",
    )

    report = verify_md_run(log_path=SUCCESS_LOG, events_path=bad_events, expected_events=1)

    assert report.ok is False
    assert report.status == MDRunStatus.REJECTED
    assert "deposited_energy_exceeds_incident_energy:evt-bad" in report.errors


def test_verify_md_run_cli_reports_verified_dataset() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "verify_md_run.py"),
            "--log",
            str(SUCCESS_LOG),
            "--events",
            str(EVENTS),
            "--expected-events",
            "2",
            "--required-ion",
            "Ar",
            "--required-material",
            "Si",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_verified=true" in result.stdout
    assert "event_count=2" in result.stdout
    assert "total_deposited_energy_eV=93.0" in result.stdout


def test_verify_md_run_cli_reports_failed_log() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "verify_md_run.py"),
            "--log",
            str(FAILED_LOG),
            "--events",
            str(EVENTS),
            "--expected-events",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "md_verified=false" in result.stdout
    assert "lammps_lost_atoms" in result.stdout
