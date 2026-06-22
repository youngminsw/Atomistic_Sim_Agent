from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_run_demo_generates_full_offline_3d_hole_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo-hole"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_demo.py"),
            "--demo",
            "pr_hole_3d",
            "--steps",
            "5",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    hit_history = json.loads((out_dir / "hit_history.json").read_text(encoding="utf-8"))
    click_index = json.loads((out_dir / "click_index.json").read_text(encoding="utf-8"))
    qa_report = json.loads((out_dir / "qa_report.json").read_text(encoding="utf-8"))
    surrogate_gate = json.loads((out_dir / "surrogate_training_gate_report.json").read_text(encoding="utf-8"))
    surrogate_manifest = json.loads((out_dir / "surrogate_model_manifest.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "demo_complete=true" in result.stdout
    assert "profile_timeline_written=true" in result.stdout
    assert "click_index_written=true" in result.stdout
    assert "qa_report_written=true" in result.stdout
    assert manifest["run_status"] == "complete"
    assert manifest["mode"] == "3d"
    assert manifest["process"]["duration_min"] == 10.0
    assert manifest["process"]["sampling_policy"] == "regular_time_interval_weighted_ions"
    assert manifest["process"]["physical_incident_count"] > manifest["ion_count"]
    assert timeline["states"][-1]["time_s"] == 600.0
    assert timeline["process"]["regular_ion_interval_s"] == 75.0
    assert hit_history["hits"][1]["time_s"] == 75.0
    assert click_index["process"]["duration_s"] == 600.0
    assert qa_report["status"] == "pass"
    assert qa_report["hard_blockers"] == []
    assert (out_dir / "transport_field.json").exists()
    assert (out_dir / "hit_history.json").exists()
    assert (out_dir / "empirical_mdn_model.json").exists()
    assert click_index["click_count"] > 1
    assert len({(click["ix"], click["iy"], click["iz"]) for click in click_index["clicks"]}) > 1
    assert manifest["surrogate"]["training_backend"] == "empirical_gaussian_mdn"
    assert manifest["surrogate"]["registered_for_feature_scale"] is True
    assert manifest["artifacts"]["surrogate_model"] == "empirical_mdn_model.json"
    assert surrogate_gate["accepted"] is True
    assert surrogate_manifest["quality_gate"]["decision"] == "accepted_for_feature_scale"


def test_run_demo_generates_full_offline_2d_trench_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo-trench"
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_demo.py"),
            "--demo",
            "pr_trench_2d",
            "--steps",
            "4",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    click_index = json.loads((out_dir / "click_index.json").read_text(encoding="utf-8"))
    qa_report = json.loads((out_dir / "qa_report.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "demo_complete=true" in result.stdout
    assert "qa_report_written=true" in result.stdout
    assert manifest["mode"] == "2d"
    assert manifest["feature_type"] == "trench"
    assert manifest["process"]["duration_min"] == 10.0
    assert manifest["surrogate"]["training_backend"] == "empirical_gaussian_mdn"
    assert timeline["states"][-1]["time_s"] == 600.0
    assert click_index["clicks"][0]["profile_history_nm"][-1] > 0.0
    assert qa_report["status"] == "pass"


def test_physics_reality_report_labels_evidence_levels(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo-hole"
    subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_demo.py"),
            "--demo",
            "pr_hole_3d",
            "--steps",
            "5",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "physics_reality_report.py"),
            "--run",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_derived=fixture_md_events" in result.stdout
    assert "surrogate_derived=empirical_gaussian_mdn" in result.stdout
    assert "surrogate_gate_accepted=true" in result.stdout
    assert "surrogate_registered_for_feature_scale=true" in result.stdout
    assert "literature_derived=seeded_provenance_registry" in result.stdout
    assert "semi_empirical=transport_sampling_and_level_set_fixture" in result.stdout
    assert "assumptions=offline_fixture_not_physical_validation" in result.stdout
    assert "first_scope_chemistry=false" in result.stdout
    assert "missing_md_coverage=true" in result.stdout


def test_reality_audit_confirms_core_policy_gates(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo-hole"
    subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_demo.py"),
            "--demo",
            "pr_hole_3d",
            "--steps",
            "5",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "physics_reality_report.py"),
            "--run",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert "model_provider_config=true" in result.stdout
    assert "gpu_host_allowlist=true" in result.stdout
    assert "neo4j_write_gate=true" in result.stdout
    assert "controlled_event_probe_active_learning_only=true" in result.stdout
