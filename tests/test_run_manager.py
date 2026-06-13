from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"
SCENE_3D = FIXTURE_ROOT / "scenes" / "pr_hole_scene.json"
IMAGE_2D = FIXTURE_ROOT / "geometry" / "pr_trench.png"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_offline_run_manager_writes_3d_hole_manifest_artifacts_and_click_index(tmp_path: Path) -> None:
    from sim_agent.runner import OfflineRunRequest, run_offline_simulation

    result = run_offline_simulation(
        OfflineRunRequest(
            run_id="offline-hole",
            mode="3d",
            source_root=SOURCE_ROOT,
            output_dir=tmp_path,
            scene_path=SCENE_3D,
            image_path=None,
            kernel_path=KERNEL,
            events_path=EVENTS,
            time_steps=5,
            ion_count=8,
            seed=7,
        )
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    click_index = json.loads(result.click_index_path.read_text(encoding="utf-8"))
    uncertainty_map = json.loads((tmp_path / "uncertainty_map.json").read_text(encoding="utf-8"))
    active_learning = json.loads((tmp_path / "active_learning_plan.json").read_text(encoding="utf-8"))
    qa_report = json.loads((tmp_path / "qa_report.json").read_text(encoding="utf-8"))

    assert result.run_status == "complete"
    assert result.artifact_count == 11
    assert manifest["run_status"] == "complete"
    assert manifest["mode"] == "3d"
    assert manifest["feature_type"] == "hole"
    assert manifest["artifacts"]["profile_timeline"] == "timeline.json"
    assert manifest["artifacts"]["uncertainty_map"] == "uncertainty_map.json"
    assert manifest["artifacts"]["active_learning_plan"] == "active_learning_plan.json"
    assert manifest["artifacts"]["surrogate_model"] == "empirical_mdn_model.json"
    assert manifest["surrogate"]["training_backend"] == "empirical_gaussian_mdn"
    assert manifest["surrogate"]["registered_for_feature_scale"] is True
    assert manifest["artifacts"]["qa_report"] == "qa_report.json"
    assert result.timeline_path.exists()
    assert result.transport_field_path.exists()
    assert result.hit_history_path.exists()
    assert result.uncertainty_map_path.exists()
    assert result.active_learning_plan_path.exists()
    assert result.qa_report_path.exists()
    assert uncertainty_map["run_id"] == "offline-hole"
    assert active_learning["controlled_event_probe_allowed"] is False
    assert active_learning["batch_size"] == 0
    assert qa_report["agent_id"] == "qa_agent"
    assert qa_report["report_version"] == "demo_qa_agent_report_v1"
    assert qa_report["evidence_scope"] == "offline_demo_fixture"
    assert qa_report["status"] == "pass"
    assert qa_report["hard_blockers"] == []
    assert "level_set_profile_timeline" in {item["check_id"] for item in qa_report["checks"]}
    assert click_index["click_count"] > 1
    assert len({(click["ix"], click["iy"], click["iz"]) for click in click_index["clicks"]}) > 1
    assert click_index["clicks"][0]["energy_transfer_eV"] > 0.0
    assert click_index["clicks"][0]["profile_history_nm"][-1] > 0.0
    assert "uncertainty_ood" in click_index["clicks"][0]
    incident = click_index["clicks"][0]["incident_history"][0]
    assert incident["event_id"] in click_index["clicks"][0]["event_ids"]
    assert incident["energy_eV"] > 0.0
    assert 0.0 <= incident["polar_deg"] <= 90.0
    assert 0.0 <= incident["azimuth_deg"] <= 360.0
    assert incident["deposited_energy_eV"] > 0.0


def test_offline_run_manager_writes_active_learning_probe_for_ood_azimuth(tmp_path: Path) -> None:
    from sim_agent.runner import OfflineRunRequest, run_offline_simulation
    from sim_agent.schemas.distributions import IonAngularDistribution

    result = run_offline_simulation(
        OfflineRunRequest(
            run_id="offline-azimuth-ood",
            mode="3d",
            source_root=SOURCE_ROOT,
            output_dir=tmp_path,
            scene_path=SCENE_3D,
            image_path=None,
            kernel_path=KERNEL,
            events_path=EVENTS,
            time_steps=3,
            ion_count=4,
            seed=7,
            angular_distribution=IonAngularDistribution(
                kind="uniform",
                polar_min_deg=30.0,
                polar_max_deg=30.0,
                azimuth_min_deg=20.0,
                azimuth_max_deg=20.0,
            ),
        )
    )

    active_learning = json.loads(result.active_learning_plan_path.read_text(encoding="utf-8"))

    assert result.run_status == "complete"
    assert active_learning["controlled_event_probe_allowed"] is True
    assert active_learning["batch_size"] == 1
    assert active_learning["requests"][0]["protocol"] == "controlled_event_probe"
    assert active_learning["requests"][0]["azimuth_range_deg"] == [20.0, 20.0]


def test_offline_run_manager_writes_2d_trench_image_run(tmp_path: Path) -> None:
    from sim_agent.runner import OfflineRunRequest, run_offline_simulation

    result = run_offline_simulation(
        OfflineRunRequest(
            run_id="offline-trench-2d",
            mode="2d",
            source_root=SOURCE_ROOT,
            output_dir=tmp_path,
            scene_path=None,
            image_path=IMAGE_2D,
            kernel_path=KERNEL,
            events_path=EVENTS,
            time_steps=4,
            ion_count=6,
            seed=5,
        )
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    click_index = json.loads(result.click_index_path.read_text(encoding="utf-8"))

    assert result.run_status == "complete"
    assert manifest["mode"] == "2d"
    assert manifest["feature_type"] == "trench"
    assert click_index["clicks"][0]["material_id"] == "Si"
    assert click_index["clicks"][0]["removed_depth_nm"] > 0.0


def test_offline_run_manager_marks_missing_kernel_failed(tmp_path: Path) -> None:
    from sim_agent.runner import OfflineRunRequest, run_offline_simulation

    result = run_offline_simulation(
        OfflineRunRequest(
            run_id="failed-kernel",
            mode="3d",
            source_root=SOURCE_ROOT,
            output_dir=tmp_path,
            scene_path=SCENE_3D,
            image_path=None,
            kernel_path=tmp_path / "missing.json",
            events_path=EVENTS,
            time_steps=1,
            ion_count=2,
            seed=1,
        )
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.run_status == "failed"
    assert result.reason == "kernel_not_found"
    assert manifest["run_status"] == "failed"
    assert manifest["reason"] == "kernel_not_found"


def test_run_offline_simulation_cli_reports_complete_3d_hole_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "hole-run"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_offline_simulation.py"),
            "--scene",
            str(SCENE_3D),
            "--kernel",
            str(KERNEL),
            "--events",
            str(EVENTS),
            "--steps",
            "5",
            "--ions",
            "8",
            "--out",
            str(out_dir),
            "--run-id",
            "cli-hole-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_status=complete" in result.stdout
    assert "profile_timeline_written=true" in result.stdout
    assert "click_index_written=true" in result.stdout
    assert "uncertainty_map_written=true" in result.stdout
    assert "active_learning_plan_written=true" in result.stdout
    assert "qa_report_written=true" in result.stdout
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "uncertainty_map.json").exists()
    assert (out_dir / "active_learning_plan.json").exists()
    assert (out_dir / "qa_report.json").exists()


def test_run_offline_simulation_cli_marks_missing_kernel_failed(tmp_path: Path) -> None:
    out_dir = tmp_path / "failed-run"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_offline_simulation.py"),
            "--scene",
            str(SCENE_3D),
            "--kernel",
            str(tmp_path / "missing.json"),
            "--events",
            str(EVENTS),
            "--steps",
            "1",
            "--ions",
            "2",
            "--out",
            str(out_dir),
            "--run-id",
            "cli-failed-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert "run_status=failed" in result.stdout
    assert "reason=kernel_not_found" in result.stdout
    assert manifest["run_status"] == "failed"
