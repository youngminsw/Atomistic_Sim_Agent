from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
SUCCESS_LOG = FIXTURE_ROOT / "md_logs" / "success_lammps.log"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"
SCENE = FIXTURE_ROOT / "scenes" / "pr_hole_scene.json"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _dataset():
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_training_dataset

    report = verify_md_run(SUCCESS_LOG, EVENTS, expected_events=2, required_ion="Ar", required_material="Si")
    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    return build_training_dataset(report, spec)


def _geometry():
    from sim_agent.geometry import GridShape, load_pattern_geometry_from_scene

    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    return load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)


def test_kmc_accumulates_position_resolved_energy_without_mutating_geometry() -> None:
    from sim_agent.kmc import IonImpact, accumulate_energy_deposition

    geometry = _geometry()
    manifest_before = geometry.export_manifest()

    field = accumulate_energy_deposition(
        geometry=geometry,
        dataset=_dataset(),
        impacts=(
            IonImpact("evt-0001", 0.0, 0.0, 1.0, 0),
            IonImpact("evt-0002", 5.0, 0.0, 1.0, 0),
        ),
    )
    center = field.diagnostic_at_nm(geometry, 0.0, 0.0, 1.0)

    assert geometry.export_manifest() == manifest_before
    assert field.feature_type.value == "hole"
    assert field.total_hit_count == 2
    assert field.cell_count == 2
    assert field.total_deposited_energy_eV == pytest.approx(93.0)
    assert field.total_removal_drive_nm == pytest.approx(0.035)
    assert center.material_id == "Si"
    assert center.region == "opening"
    assert center.deposited_energy_eV == pytest.approx(65.0)
    assert center.event_ids == ("evt-0001",)


def test_pr_mask_hit_uses_selectivity_scaled_removal_drive() -> None:
    from sim_agent.kmc import IonImpact, accumulate_energy_deposition

    geometry = _geometry()

    field = accumulate_energy_deposition(
        geometry=geometry,
        dataset=_dataset(),
        impacts=(IonImpact("evt-0001", 18.0, 18.0, 1.0, 0),),
    )
    diagnostic = field.diagnostic_at_nm(geometry, 18.0, 18.0, 1.0)

    assert diagnostic.material_id == "PR"
    assert diagnostic.region == "mask"
    assert diagnostic.removal_drive_nm == pytest.approx(0.03 / 20.0)
    assert diagnostic.removal_law == "mask_selectivity_scaled"


def test_unknown_impact_event_is_rejected() -> None:
    from sim_agent.kmc import IonImpact, KMCTransportError, accumulate_energy_deposition

    with pytest.raises(KMCTransportError, match="surrogate_row_missing_for_impact"):
        accumulate_energy_deposition(
            geometry=_geometry(),
            dataset=_dataset(),
            impacts=(IonImpact("evt-missing", 0.0, 0.0, 1.0, 0),),
        )


def test_build_kmc_field_cli_reports_energy_field() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_kmc_field.py"),
            "--scene",
            str(SCENE),
            "--log",
            str(SUCCESS_LOG),
            "--events",
            str(EVENTS),
            "--kernel",
            str(KERNEL),
            "--expected-events",
            "2",
            "--required-ion",
            "Ar",
            "--required-material",
            "Si",
            "--impact",
            "evt-0001:0,0,1",
            "--impact",
            "evt-0002:5,0,1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "kmc_field_ok=true" in result.stdout
    assert "feature_type=hole" in result.stdout
    assert "hit_count=2" in result.stdout
    assert "cell_count=2" in result.stdout
    assert "total_deposited_energy_eV=93.0" in result.stdout
    assert "geometry_mutated=false" in result.stdout
