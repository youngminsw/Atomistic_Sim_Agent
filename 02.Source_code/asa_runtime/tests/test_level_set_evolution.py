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


def _target_field():
    from sim_agent.kmc import IonImpact, accumulate_energy_deposition

    return accumulate_energy_deposition(
        geometry=_geometry(),
        dataset=_dataset(),
        impacts=(
            IonImpact("evt-0001", 0.0, 0.0, 1.0, 0),
            IonImpact("evt-0002", 5.0, 0.0, 1.0, 0),
        ),
    )


def test_level_set_evolves_flat_target_profile_downward_over_time() -> None:
    from sim_agent.level_set import LevelSetConfig, evolve_profile

    geometry = _geometry()
    timeline = evolve_profile(_target_field(), LevelSetConfig(time_steps=3, time_step_s=0.25, cell_area_nm2=4.0))
    diagnostic = timeline.diagnostic_at_nm(geometry, 0.0, 0.0, 1.0)

    assert timeline.state_count == 4
    assert timeline.states[0].total_removed_volume_nm3 == 0.0
    assert timeline.final_state.total_removed_volume_nm3 == pytest.approx(0.14)
    assert diagnostic.material_id == "Si"
    assert diagnostic.region == "opening"
    assert diagnostic.depth_history_nm == pytest.approx((0.0, 0.01, 0.02, 0.03))
    assert diagnostic.energy_history_eV == pytest.approx((0.0, 65.0 / 3.0, 130.0 / 3.0, 65.0))
    assert diagnostic.removal_law == "target_surrogate_direct"
    assert diagnostic.event_ids == ("evt-0001",)


def test_level_set_pr_mask_profile_evolves_with_slow_selectivity() -> None:
    from sim_agent.kmc import IonImpact, accumulate_energy_deposition
    from sim_agent.level_set import LevelSetConfig, evolve_profile

    geometry = _geometry()
    field = accumulate_energy_deposition(
        geometry=geometry,
        dataset=_dataset(),
        impacts=(IonImpact("evt-0001", 18.0, 18.0, 1.0, 0),),
    )

    timeline = evolve_profile(field, LevelSetConfig(time_steps=3, time_step_s=0.25, cell_area_nm2=4.0))
    diagnostic = timeline.diagnostic_at_nm(geometry, 18.0, 18.0, 1.0)

    assert diagnostic.material_id == "PR"
    assert diagnostic.region == "mask"
    assert diagnostic.depth_history_nm == pytest.approx((0.0, 0.0005, 0.001, 0.0015))
    assert diagnostic.removal_law == "mask_selectivity_scaled"
    assert timeline.final_state.total_removed_volume_nm3 == pytest.approx(0.006)


def test_level_set_rejects_non_positive_time_steps() -> None:
    from sim_agent.level_set import LevelSetConfig, LevelSetError

    with pytest.raises(LevelSetError, match="time_steps_must_be_positive"):
        LevelSetConfig(time_steps=0, time_step_s=0.25, cell_area_nm2=4.0)


def test_build_level_set_timeline_cli_reports_click_history() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_level_set_timeline.py"),
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
            "--time-steps",
            "3",
            "--time-step-s",
            "0.25",
            "--cell-area-nm2",
            "4.0",
            "--click-nm",
            "0,0,1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "level_set_ok=true" in result.stdout
    assert "state_count=4" in result.stdout
    assert "final_removed_volume_nm3=0.140" in result.stdout
    assert "click_material=Si" in result.stdout
    assert "click_depth_history_nm=0.000,0.010,0.020,0.030" in result.stdout
