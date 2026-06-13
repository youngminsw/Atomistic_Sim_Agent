from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
HIGH_UNCERTAINTY_RUN = FIXTURE_ROOT / "runs" / "high_uncertainty_ar_si"
CHEMISTRY_RUN = FIXTURE_ROOT / "runs" / "request_chemistry_extension"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_active_learning_plan_clusters_high_uncertainty_samples() -> None:
    from sim_agent.md_campaign import plan_active_learning_run

    plan = plan_active_learning_run(HIGH_UNCERTAINTY_RUN)

    assert plan.same_expert is True
    assert plan.batch_size == 1
    assert plan.controlled_event_probe_allowed is True
    assert plan.requests[0].protocol == "controlled_event_probe"
    assert plan.requests[0].handoff_target == "md_agent"
    assert plan.requests[0].force_field_protocol_id == "Si_Tersoff_ZBL_physical_v001"
    assert plan.requests[0].physics_scope == "physical_bombardment_no_chemistry"
    assert plan.requests[0].sample_count == 2
    assert plan.requests[0].sample_ids == ("u-001", "u-002")
    assert plan.requests[0].same_material_only is True
    assert plan.requests[0].energy_range_eV == pytest.approx((1800.0, 2000.0))
    assert plan.requests[0].snapshot_reset_required is True
    assert plan.requests[0].source_snapshot_ids == ("snap-top-001", "snap-top-002")


def test_active_learning_rejects_chemistry_extension() -> None:
    from sim_agent.md_campaign import ActiveLearningPlanError, plan_active_learning_run

    with pytest.raises(ActiveLearningPlanError, match="new_campaign_required"):
        plan_active_learning_run(CHEMISTRY_RUN)


def test_surrogate_uncertainty_map_feeds_active_learning_planner(tmp_path: Path) -> None:
    from sim_agent.md_campaign import plan_active_learning_run
    from sim_agent.ml_surrogate import (
        InteractionContext,
        KernelFeatureSpec,
        UncertaintyMapSample,
        build_fixture_interaction_kernel,
        write_uncertainty_map,
    )

    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    kernel = build_fixture_interaction_kernel(EVENTS, spec, provenance_source=str(EVENTS))
    context = InteractionContext(
        ion_species="Ar",
        material_id="Si",
        force_field_protocol_id="Si_Tersoff_ZBL_physical_v001",
        physics_scope="physical_bombardment_no_chemistry",
        energy_eV=100.0,
        polar_deg=30.0,
        azimuth_deg=20.0,
        local_incidence_deg=30.0,
        phase="crystal",
        amorphous_index=0.0,
        roughness_rms_nm=0.1,
        rdf_crystal_similarity=0.92,
        rdf_amorphous_similarity=0.08,
        damage_dose=0.0,
        implanted_inert_fraction=0.0,
        local_fluence=0.0,
        removed_depth_nm=0.0,
    )
    inference = kernel.infer(context)

    write_uncertainty_map(
        tmp_path / "uncertainty_map.json",
        "kernel-azimuth-ood",
        kernel.manifest,
        (UncertaintyMapSample("sample-azimuth-001", context, inference, "snap-azimuth-001"),),
    )
    plan = plan_active_learning_run(tmp_path)

    assert plan.run_id == "kernel-azimuth-ood"
    assert plan.controlled_event_probe_allowed is True
    assert plan.requests[0].azimuth_range_deg == pytest.approx((20.0, 20.0))
    assert plan.requests[0].source_snapshot_ids == ("snap-azimuth-001",)


def test_plan_active_learning_cli_outputs_probe_batch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_active_learning.py"),
            "--run",
            str(HIGH_UNCERTAINTY_RUN),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "active_learning_plan_ok=true" in result.stdout
    assert "same_expert=true" in result.stdout
    assert "batch_size=1" in result.stdout
    assert "controlled_event_probe_allowed=true" in result.stdout
    assert "handoff_target=md_agent" in result.stdout
    assert "same_material_only=true" in result.stdout
    assert "snapshot_reset_required=true" in result.stdout


def test_smoke_physics_kernel_cli_writes_uncertainty_map(tmp_path: Path) -> None:
    map_path = tmp_path / "uncertainty_map.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_physics_kernel.py"),
            "--fixture",
            str(EVENTS),
            "--context",
            "ar_si_100ev_30deg",
            "--azimuth-deg",
            "20",
            "--uncertainty-map-out",
            str(map_path),
            "--run-id",
            "cli-azimuth-ood",
            "--sample-id",
            "cli-sample-001",
            "--snapshot-id",
            "cli-snap-001",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert map_path.exists()

    plan_result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_active_learning.py"),
            "--run",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert plan_result.returncode == 0, plan_result.stdout + plan_result.stderr
    assert "active_learning_plan_ok=true" in plan_result.stdout
    assert "source_snapshot_ids=cli-snap-001" in plan_result.stdout


def test_plan_active_learning_cli_rejects_chemistry_extension() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_active_learning.py"),
            "--run",
            str(CHEMISTRY_RUN),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "active_learning_plan_ok=false" in result.stdout
    assert "new_campaign_required" in result.stdout
