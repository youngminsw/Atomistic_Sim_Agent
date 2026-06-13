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
FAILED_LOG = FIXTURE_ROOT / "md_logs" / "failed_lammps.log"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"


def test_train_empirical_mdn_surrogate_writes_accepted_model_manifest(tmp_path: Path) -> None:
    result = _run_training(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mdn_training_ok=true" in result.stdout
    assert "training_backend=empirical_gaussian_mdn" in result.stdout
    assert "quality_gate_decision=accepted_for_feature_scale" in result.stdout
    assert "registered_for_feature_scale=true" in result.stdout

    model = json.loads((tmp_path / "empirical_mdn_model.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "surrogate_model_manifest.json").read_text(encoding="utf-8"))

    assert model["model_type"] == "mdn_surrogate"
    assert model["training_backend"] == "empirical_gaussian_mdn"
    assert len(model["mixture_components"]) == 2
    assert manifest["model_type"] == "mdn_surrogate"
    assert manifest["model_artifact"] == "empirical_mdn_model.json"
    assert manifest["quality_gate"]["accepted"] is True
    assert manifest["quality_gate"]["max_high_uncertainty_fraction"] == 0.01


def test_train_empirical_mdn_surrogate_blocks_unverified_md(tmp_path: Path) -> None:
    result = _run_training(tmp_path, log=FAILED_LOG)

    assert result.returncode != 0
    assert "mdn_training_ok=false" in result.stdout
    assert "surrogate_training_ok=false" in result.stdout
    assert "verified_md_required" in result.stdout
    assert not (tmp_path / "empirical_mdn_model.json").exists()
    assert not (tmp_path / "surrogate_model_manifest.json").exists()


def test_train_empirical_mdn_surrogate_blocks_high_uncertainty_registry(
    tmp_path: Path,
) -> None:
    result = _run_training(tmp_path, high_uncertainty_fraction="0.4")

    assert result.returncode != 0
    assert "mdn_training_ok=false" in result.stdout
    assert "quality_gate_decision=active_learning_required" in result.stdout
    assert "next_actions=plan_active_learning_md,rerun_surrogate_training_gate" in result.stdout
    assert (tmp_path / "empirical_mdn_model.json").exists()
    assert not (tmp_path / "surrogate_model_manifest.json").exists()


def _run_training(
    output_dir: Path,
    log: Path = SUCCESS_LOG,
    high_uncertainty_fraction: str = "0.005",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "train_empirical_mdn_surrogate.py"),
            "--log",
            str(log),
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
            "--validation-event-count",
            "2",
            "--validation-nll",
            "0.08",
            "--deposited-energy-mae-eV",
            "1.0",
            "--sputter-yield-mae",
            "0.04",
            "--reflection-brier-score",
            "0.02",
            "--calibration-error",
            "0.03",
            "--high-uncertainty-fraction",
            high_uncertainty_fraction,
            "--min-training-events",
            "2",
            "--min-validation-events",
            "1",
            "--required-energy-range-eV",
            "80:100",
            "--required-polar-range-deg",
            "30:45",
            "--required-azimuth-range-deg",
            "120:240",
            "--out-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
