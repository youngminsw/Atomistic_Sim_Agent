from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_agent_cli_ledger_blocks_amorphous_md_without_structure_source(
    tmp_path: Path,
) -> None:
    # Given
    output_dir = tmp_path / "missing-amorphous-source"

    # When
    result = subprocess.run(
        _agent_cli_args(output_dir),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    ledger = as_mapping(
        json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8")),
        "agent_run_ledger",
    )
    md = as_mapping(ledger["md"], "md")
    qa = as_mapping(ledger["qa"], "qa")
    assert result.returncode == 0, result.stdout + result.stderr
    assert md["incident_count"] == 500
    assert md["phase"] == "amorphous"
    assert md["production_ready"] is False
    assert "amorphous_lammps_structure_source_required" in md["hard_blockers"]
    assert qa["status"] == "blocked"
    assert "amorphous_lammps_structure_source_required" in qa["hard_blockers"]


def test_agent_cli_ledger_accepts_user_supplied_amorphous_structure_source(
    tmp_path: Path,
) -> None:
    # Given
    output_dir = tmp_path / "ready-amorphous-source"
    source = tmp_path / "a_si_relaxed.data"
    source.write_text("5000 atoms\n", encoding="utf-8")

    # When
    result = subprocess.run(
        _agent_cli_args(output_dir)
        + [
            "--lammps-structure-source",
            source.as_uri(),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    ledger = as_mapping(
        json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8")),
        "agent_run_ledger",
    )
    md = as_mapping(ledger["md"], "md")
    qa = as_mapping(ledger["qa"], "qa")
    assert result.returncode == 0, result.stdout + result.stderr
    assert md["production_ready"] is True
    assert md["structure_source_present"] is True
    assert md["hard_blockers"] == []
    assert qa["status"] == "pass"


def test_md_production_acceptance_blocks_verified_but_too_small_run() -> None:
    # Given
    from sim_agent.md import assess_md_production_acceptance

    postprocess_report: JsonMap = {
        "postprocess_status": "md_postprocess_complete",
        "ok": True,
        "event_count": 2,
        "expected_incident_count": 2,
        "layer_removed_count": 1,
        "total_deposited_energy_eV": 93.0,
        "verification_status": "verified",
        "evidence": ["md_events_parsed", "lammps_completed"],
        "errors": [],
    }

    # When
    report = assess_md_production_acceptance(postprocess_report)

    # Then
    assert report.accepted is False
    assert report.payload["minimum_incidents"] == 500
    assert "event_count_too_low:2<500" in report.payload["blockers"]
    assert "expected_incident_count_too_low:2<500" in report.payload["blockers"]


def test_validate_md_production_acceptance_cli_accepts_500_event_report(
    tmp_path: Path,
) -> None:
    # Given
    report_path = tmp_path / "md_postprocess_report.json"
    out_path = tmp_path / "md_production_acceptance.json"
    report_path.write_text(json.dumps(_accepted_postprocess_report()), encoding="utf-8")

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "validate_md_production_acceptance.py"),
            "--postprocess-report",
            str(report_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "acceptance")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_production_acceptance=true" in result.stdout
    assert payload["accepted"] is True
    assert "md_500_incidents_verified" in payload["evidence"]


def test_validate_md_production_acceptance_cli_blocks_small_report(
    tmp_path: Path,
) -> None:
    # Given
    report_path = tmp_path / "md_postprocess_report.json"
    out_path = tmp_path / "md_production_acceptance.json"
    report_path.write_text(json.dumps(_small_postprocess_report()), encoding="utf-8")

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "validate_md_production_acceptance.py"),
            "--postprocess-report",
            str(report_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "acceptance")
    assert result.returncode == 1
    assert "md_production_acceptance=false" in result.stdout
    assert "event_count_too_low:2<500" in payload["blockers"]


def _agent_cli_args(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
        "--offline",
        "--goal",
        "Plan Ar etching on amorphous Si through the production MD pipeline",
        "--material",
        "Si",
        "--phase",
        "amorphous",
        "--ion",
        "Ar",
        "--feature-type",
        "hole",
        "--energy-range-eV",
        "30:150",
        "--polar-range-deg",
        "0:55",
        "--azimuth-range-deg",
        "0:360",
        "--host",
        "gpu-5090",
        "--environment-name",
        "atomistic-sim-gpu",
        "--output-dir",
        str(output_dir),
    ]


def _small_postprocess_report() -> JsonMap:
    return {
        "postprocess_status": "md_postprocess_complete",
        "ok": True,
        "event_count": 2,
        "expected_incident_count": 2,
        "layer_removed_count": 1,
        "total_deposited_energy_eV": 93.0,
        "verification_status": "verified",
        "evidence": ["md_events_parsed", "lammps_completed"],
        "errors": [],
    }


def _accepted_postprocess_report() -> JsonMap:
    return _small_postprocess_report() | {
        "event_count": 500,
        "expected_incident_count": 500,
        "total_deposited_energy_eV": 18_500.0,
    }
