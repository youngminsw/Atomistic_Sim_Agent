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
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"
SCENE_3D = FIXTURE_ROOT / "scenes" / "pr_hole_scene.json"
IMAGE_2D = FIXTURE_ROOT / "geometry" / "pr_trench.png"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _registry():
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_fixture_interaction_kernel

    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    return build_fixture_interaction_kernel(EVENTS, spec, provenance_source=str(EVENTS)).registry()


def _energy_distribution():
    from sim_agent.schemas.distributions import IonEnergyBin, IonEnergyDistribution

    return IonEnergyDistribution(
        kind="histogram",
        unit="eV",
        bins=(
            IonEnergyBin(min=80.0, max=90.0, probability=0.5),
            IonEnergyBin(min=90.0, max=100.0, probability=0.5),
        ),
    )


def _angular_distribution():
    from sim_agent.schemas.distributions import IonAngularDistribution

    return IonAngularDistribution(
        kind="uniform",
        polar_min_deg=30.0,
        polar_max_deg=45.0,
        azimuth_min_deg=120.0,
        azimuth_max_deg=240.0,
    )


def test_transport_sampler_draws_from_iedf_and_iadf() -> None:
    from sim_agent.transport import sample_ions

    samples = sample_ions(_energy_distribution(), _angular_distribution(), ion_count=8, seed=7)

    assert len(samples) == 8
    assert all(80.0 <= sample.energy_eV <= 100.0 for sample in samples)
    assert all(30.0 <= sample.polar_deg <= 45.0 for sample in samples)
    assert all(120.0 <= sample.azimuth_deg <= 240.0 for sample in samples)
    assert tuple(sample.time_step for sample in samples) == tuple(range(8))
    assert tuple(sample.time_s for sample in samples) == pytest.approx(tuple(float(index) for index in range(8)))


def test_transport_sampler_uses_regular_process_time_intervals() -> None:
    from sim_agent.transport import sample_ions

    samples = sample_ions(
        _energy_distribution(),
        _angular_distribution(),
        ion_count=4,
        seed=7,
        duration_s=600.0,
    )

    assert tuple(sample.time_s for sample in samples) == pytest.approx((0.0, 150.0, 300.0, 450.0))


def test_transport_3d_hole_accumulates_energy_damage_and_history_without_mutation() -> None:
    from sim_agent.geometry import GridShape, load_pattern_geometry_from_scene
    from sim_agent.transport import run_transport_3d

    scene = json.loads(SCENE_3D.read_text(encoding="utf-8"))
    geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
    manifest_before = geometry.export_manifest()

    result = run_transport_3d(
        geometry=geometry,
        registry=_registry(),
        energy_distribution=_energy_distribution(),
        angular_distribution=_angular_distribution(),
        ion_count=4,
        seed=11,
    )

    first_cell = result.field.cells[0]

    assert geometry.export_manifest() == manifest_before
    assert result.mode == "3d"
    assert result.feature_type == "hole"
    assert result.hit_history_count == 4
    assert result.field.cell_count > 1
    assert result.field.total_deposited_energy_eV > 0.0
    assert result.field.total_removed_depth_nm > 0.0
    assert first_cell.local_fluence == pytest.approx(float(first_cell.hit_count))
    assert first_cell.damage_dose > 0.0
    assert first_cell.roughness_rms_nm > 0.1
    assert first_cell.implanted_inert_fraction >= 0.0
    assert result.hit_history[0].time_step == 0
    assert result.hit_history[0].time_s == pytest.approx(0.0)
    assert result.hit_history[0].local_incidence_deg == pytest.approx(result.hit_history[0].polar_deg)
    assert len({(hit.x_nm, hit.y_nm) for hit in result.hit_history}) > 1


def test_transport_2d_trench_produces_time_position_indexed_history() -> None:
    from sim_agent.geometry import load_pattern_geometry_2d
    from sim_agent.transport import run_transport_2d

    geometry = load_pattern_geometry_2d(
        IMAGE_2D,
        pixel_size_nm=1.0,
        target_material_id="Si",
        mask_material_id="PR",
        structure_description="2D trench transport fixture",
    )

    result = run_transport_2d(
        geometry=geometry,
        registry=_registry(),
        energy_distribution=_energy_distribution(),
        angular_distribution=_angular_distribution(),
        ion_count=3,
        seed=5,
    )

    assert result.mode == "2d"
    assert result.feature_type == "trench"
    assert result.hit_history_count == 3
    assert result.field.cell_count > 1
    assert result.hit_history[0].z_nm == pytest.approx(0.0)
    assert result.hit_history[0].material_id == "Si"
    assert len({hit.x_nm for hit in result.hit_history}) > 1


def test_local_incidence_angle_uses_surface_normal() -> None:
    from sim_agent.transport import SurfaceNormal3D, local_incidence_angle_deg

    flat = local_incidence_angle_deg(30.0, 0.0, SurfaceNormal3D(0.0, 0.0, -1.0))
    sidewall = local_incidence_angle_deg(30.0, 0.0, SurfaceNormal3D(1.0, 0.0, 0.0))

    assert flat == pytest.approx(30.0)
    assert sidewall == pytest.approx(60.0)


def test_smoke_transport_cli_reports_3d_hole_transport() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_transport.py"),
            "--scene",
            str(SCENE_3D),
            "--kernel",
            str(KERNEL),
            "--events",
            str(EVENTS),
            "--ions",
            "6",
            "--seed",
            "7",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "transport_valid=true" in result.stdout
    assert "feature_type=hole" in result.stdout
    assert "energy_field_written=true" in result.stdout
    assert "hit_history_count=6" in result.stdout
    assert "geometry_mutated=false" in result.stdout


def test_smoke_transport_cli_blocks_missing_kernel() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_transport.py"),
            "--scene",
            str(SCENE_3D),
            "--kernel",
            "missing.json",
            "--events",
            str(EVENTS),
            "--ions",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "transport_valid=false" in result.stdout
    assert "kernel_not_found" in result.stdout
