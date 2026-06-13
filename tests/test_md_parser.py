from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
VALID_RUN = FIXTURE_ROOT / "md_runs" / "small_valid"
FAILED_RUN = FIXTURE_ROOT / "md_runs" / "failed_log"
MATERIAL_ROOT = FIXTURE_ROOT / "materials"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_parse_lammps_output_run_emits_events_with_descriptors(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    out_path = tmp_path / "md_events.jsonl"
    report = parse_lammps_output_run(
        run_dir=VALID_RUN,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=out_path,
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert report.ok is True
    assert report.event_count == 2
    assert report.descriptors_present is True
    assert report.layer_removed_count == 1
    assert report.total_deposited_energy_eV == pytest.approx(93.0)
    assert rows[0]["event_id"] == "evt-0001"
    assert rows[0]["pre_state"]["rdf_order_features"]["crystal_similarity"] == pytest.approx(0.92)
    assert rows[0]["post_delta"]["roughness_rms_nm"] == pytest.approx(0.14)
    assert rows[0]["outcome"]["event_type"] == "sputter"
    assert rows[1]["layer_removed"] is True
    assert rows[1]["post_delta"]["removed_depth_nm"] == pytest.approx(0.005)


def test_parse_lammps_output_run_rejects_failed_log(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    report = parse_lammps_output_run(
        run_dir=FAILED_RUN,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=tmp_path / "events.jsonl",
    )

    assert report.ok is False
    assert report.errors == ("lammps_not_successful",)


def test_parse_lammps_output_run_rejects_missing_required_dump(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    run_dir = tmp_path / "missing_dump"
    shutil.copytree(VALID_RUN, run_dir)
    (run_dir / "sputtered.dump").unlink()

    report = parse_lammps_output_run(
        run_dir=run_dir,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=tmp_path / "events.jsonl",
    )

    assert report.ok is False
    assert report.errors == ("missing:sputtered.dump",)


def test_parse_lammps_output_run_rejects_unphysical_reflection_energy(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    run_dir = tmp_path / "bad_reflection_energy"
    shutil.copytree(VALID_RUN, run_dir)
    (run_dir / "reflected.dump").write_text(
        "event_id,reflected,energy_out_eV,polar_deg,azimuth_deg\n"
        "evt-0001,true,120.0,30.0,120.0\n"
        "evt-0002,true,43.0,55.0,250.0\n",
        encoding="utf-8",
    )

    report = parse_lammps_output_run(
        run_dir=run_dir,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=tmp_path / "events.jsonl",
    )

    assert report.ok is False
    assert report.errors == ("reflected_energy_exceeds_incident_energy:evt-0001",)


def test_parse_lammps_output_run_rejects_invalid_implant_fraction(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    run_dir = tmp_path / "bad_implant_fraction"
    shutil.copytree(VALID_RUN, run_dir)
    (run_dir / "implanted.dump").write_text(
        "event_id,retained_fraction,depth_mean_nm\n"
        "evt-0001,1.2,0.0\n"
        "evt-0002,0.15,1.2\n",
        encoding="utf-8",
    )

    report = parse_lammps_output_run(
        run_dir=run_dir,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=tmp_path / "events.jsonl",
    )

    assert report.ok is False
    assert report.errors == ("implant_retained_fraction_out_of_range:evt-0001",)


def test_parse_lammps_output_run_rejects_invalid_incident_angle(tmp_path: Path) -> None:
    from sim_agent.md import parse_lammps_output_run

    run_dir = tmp_path / "bad_incident_angle"
    shutil.copytree(VALID_RUN, run_dir)
    (run_dir / "incident.dump").write_text(
        "event_id,ion,material_id,energy_eV,polar_deg,azimuth_deg\n"
        "evt-0001,Ar,Si,100.0,100.0,120.0\n"
        "evt-0002,Ar,Si,80.0,45.0,240.0\n",
        encoding="utf-8",
    )

    report = parse_lammps_output_run(
        run_dir=run_dir,
        material_id="Si",
        descriptor_root=MATERIAL_ROOT,
        out_path=tmp_path / "events.jsonl",
    )

    assert report.ok is False
    assert report.errors == ("incident_polar_deg_out_of_range:evt-0001",)


def test_parse_md_events_cli_outputs_valid_event_dataset(tmp_path: Path) -> None:
    out_path = tmp_path / "events.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "parse_md_events.py"),
            "--fixture",
            str(VALID_RUN),
            "--material",
            "Si",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_events_valid=true" in result.stdout
    assert "event_count=2" in result.stdout
    assert "descriptors_present=true" in result.stdout
    assert "layer_removed_count=1" in result.stdout
    assert out_path.exists()


def test_parse_md_events_cli_rejects_failed_lammps_log(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "parse_md_events.py"),
            "--fixture",
            str(FAILED_RUN),
            "--material",
            "Si",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--out",
            str(tmp_path / "events.jsonl"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "md_events_valid=false" in result.stdout
    assert "lammps_not_successful" in result.stdout
