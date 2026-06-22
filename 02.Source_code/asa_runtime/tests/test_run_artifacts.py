from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
SUCCESS_LOG = FIXTURE_ROOT / "md_logs" / "success_lammps.log"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"
SCENE = FIXTURE_ROOT / "scenes" / "pr_hole_scene.json"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _geometry():
    from sim_agent.geometry import GridShape, load_pattern_geometry_from_scene

    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    return load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)


def _timeline():
    from sim_agent.kmc import IonImpact, accumulate_energy_deposition
    from sim_agent.level_set import LevelSetConfig, evolve_profile
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_training_dataset

    report = verify_md_run(SUCCESS_LOG, EVENTS, expected_events=2, required_ion="Ar", required_material="Si")
    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    dataset = build_training_dataset(report, spec)
    field = accumulate_energy_deposition(
        geometry=_geometry(),
        dataset=dataset,
        impacts=(
            IonImpact("evt-0001", 0.0, 0.0, 1.0, 0),
            IonImpact("evt-0002", 5.0, 0.0, 1.0, 0),
        ),
    )
    return evolve_profile(field, LevelSetConfig(time_steps=3, time_step_s=0.25, cell_area_nm2=4.0))


def test_write_profile_run_bundle_saves_ui_readable_artifacts(tmp_path: Path) -> None:
    from sim_agent.run_artifacts import write_profile_run_bundle

    bundle = write_profile_run_bundle(
        output_dir=tmp_path,
        run_id="fixture-run",
        geometry=_geometry(),
        timeline=_timeline(),
        click_points_nm=((0.0, 0.0, 1.0), (5.0, 0.0, 1.0)),
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))

    assert bundle.manifest_path == tmp_path / "manifest.json"
    assert bundle.timeline_path == tmp_path / "timeline.json"
    assert bundle.diagnostics_path == tmp_path / "diagnostics.json"
    assert manifest["run_id"] == "fixture-run"
    assert manifest["feature_type"] == "hole"
    assert manifest["artifact_types"] == ["run_manifest", "profile_timeline", "click_diagnostics"]
    assert timeline["state_count"] == 4
    assert timeline["final_removed_volume_nm3"] == 0.14
    assert timeline["states"][3]["cells"][0]["surface_depth_nm"] == 0.03
    assert diagnostics["click_count"] == 2
    assert diagnostics["clicks"][0]["depth_history_nm"] == [0.0, 0.01, 0.02, 0.03]
    assert diagnostics["clicks"][0]["energy_history_eV"] == [0.0, 21.666667, 43.333333, 65.0]


def test_write_profile_run_bundle_rejects_empty_click_points(tmp_path: Path) -> None:
    from sim_agent.run_artifacts import RunArtifactError, write_profile_run_bundle

    try:
        write_profile_run_bundle(
            output_dir=tmp_path,
            run_id="fixture-run",
            geometry=_geometry(),
            timeline=_timeline(),
            click_points_nm=(),
        )
    except RunArtifactError as exc:
        assert str(exc) == "click_points_required"
    else:
        raise AssertionError("expected RunArtifactError")


def test_write_run_bundle_cli_reports_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "write_run_bundle.py"),
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
            "--output-dir",
            str(output_dir),
            "--run-id",
            "cli-fixture-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_bundle_ok=true" in result.stdout
    assert "run_id=cli-fixture-run" in result.stdout
    assert "artifact_count=3" in result.stdout
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "timeline.json").exists()
    assert (output_dir / "diagnostics.json").exists()
