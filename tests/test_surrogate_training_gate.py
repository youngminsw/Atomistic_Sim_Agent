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

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _kernel():
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_fixture_interaction_kernel

    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    return build_fixture_interaction_kernel(EVENTS, spec, provenance_source=str(EVENTS))


def _criteria():
    from sim_agent.ml_surrogate import (
        CoverageRange,
        KernelCoverage,
        SurrogateTrainingCriteria,
    )

    return SurrogateTrainingCriteria(
        min_training_events=2,
        min_validation_events=1,
        max_validation_nll=0.25,
        max_deposited_energy_mae_eV=5.0,
        max_sputter_yield_mae=0.2,
        max_reflection_brier_score=0.1,
        max_calibration_error=0.08,
        max_high_uncertainty_fraction=0.01,
        required_coverage=KernelCoverage(
            energy_eV=CoverageRange(80.0, 100.0),
            polar_deg=CoverageRange(30.0, 45.0),
            azimuth_deg=CoverageRange(120.0, 240.0),
        ),
    )


def _good_metrics():
    from sim_agent.ml_surrogate import MDNTrainingMetrics

    return MDNTrainingMetrics(
        validation_event_count=2,
        validation_nll=0.08,
        deposited_energy_mae_eV=1.2,
        sputter_yield_mae=0.05,
        reflection_brier_score=0.03,
        calibration_error=0.02,
        high_uncertainty_fraction=0.005,
    )


def test_surrogate_training_gate_accepts_validated_mdn_for_feature_scale() -> None:
    from sim_agent.ml_surrogate import assess_surrogate_training_readiness

    report = assess_surrogate_training_readiness(
        _kernel().manifest,
        _good_metrics(),
        _criteria(),
    )

    assert report.accepted is True
    assert report.decision == "accepted_for_feature_scale"
    assert "feature_space_coverage_sufficient" in report.evidence
    assert "validation_metrics_within_thresholds" in report.evidence


def test_surrogate_training_gate_requests_active_learning_for_uncertain_regions() -> None:
    from sim_agent.ml_surrogate import MDNTrainingMetrics, assess_surrogate_training_readiness

    metrics = MDNTrainingMetrics(
        validation_event_count=2,
        validation_nll=0.09,
        deposited_energy_mae_eV=1.0,
        sputter_yield_mae=0.04,
        reflection_brier_score=0.02,
        calibration_error=0.03,
        high_uncertainty_fraction=0.4,
    )

    report = assess_surrogate_training_readiness(_kernel().manifest, metrics, _criteria())

    assert report.accepted is False
    assert report.decision == "active_learning_required"
    assert "high_uncertainty_fraction_too_high" in report.blockers
    assert "plan_active_learning_md" in report.next_actions


def test_surrogate_training_gate_retrains_underfit_mdn_before_use() -> None:
    from sim_agent.ml_surrogate import MDNTrainingMetrics, assess_surrogate_training_readiness

    metrics = MDNTrainingMetrics(
        validation_event_count=2,
        validation_nll=0.8,
        deposited_energy_mae_eV=20.0,
        sputter_yield_mae=0.8,
        reflection_brier_score=0.4,
        calibration_error=0.2,
        high_uncertainty_fraction=0.05,
    )

    report = assess_surrogate_training_readiness(_kernel().manifest, metrics, _criteria())

    assert report.accepted is False
    assert report.decision == "retrain_required"
    assert "validation_nll_too_high" in report.blockers
    assert "retrain_mdn" in report.next_actions


def test_surrogate_training_gate_cli_writes_agent_action_report(tmp_path: Path) -> None:
    report_path = tmp_path / "training_gate_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "assess_surrogate_training_gate.py"),
            "--events",
            str(EVENTS),
            "--kernel",
            str(KERNEL),
            "--validation-event-count",
            "2",
            "--validation-nll",
            "0.09",
            "--deposited-energy-mae-eV",
            "1.0",
            "--sputter-yield-mae",
            "0.04",
            "--reflection-brier-score",
            "0.02",
            "--calibration-error",
            "0.03",
            "--high-uncertainty-fraction",
            "0.4",
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
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert "surrogate_training_gate_ok=false" in result.stdout
    assert "decision=active_learning_required" in result.stdout
    assert payload["decision"] == "active_learning_required"
    assert "plan_active_learning_md" in payload["next_actions"]


def test_surrogate_model_registry_requires_accepted_gate(tmp_path: Path) -> None:
    from sim_agent.ml_surrogate import (
        MDNTrainingMetrics,
        SurrogateModelRegistryError,
        assess_surrogate_training_readiness,
        register_surrogate_model,
    )

    metrics = MDNTrainingMetrics(
        validation_event_count=2,
        validation_nll=0.09,
        deposited_energy_mae_eV=1.0,
        sputter_yield_mae=0.04,
        reflection_brier_score=0.02,
        calibration_error=0.03,
        high_uncertainty_fraction=0.4,
    )
    gate = assess_surrogate_training_readiness(_kernel().manifest, metrics, _criteria())

    try:
        register_surrogate_model(tmp_path, _kernel().manifest, metrics, gate, "mdn.pt")
    except SurrogateModelRegistryError as exc:
        assert str(exc) == "surrogate_gate_not_accepted"
    else:
        raise AssertionError("expected rejected registry write")


def test_surrogate_model_registry_writes_accepted_mdn_manifest(tmp_path: Path) -> None:
    from sim_agent.ml_surrogate import (
        assess_surrogate_training_readiness,
        register_surrogate_model,
    )

    kernel = _kernel()
    metrics = _good_metrics()
    gate = assess_surrogate_training_readiness(kernel.manifest, metrics, _criteria())

    registration = register_surrogate_model(tmp_path, kernel.manifest, metrics, gate, "mdn.pt")

    payload = json.loads(registration.registry_path.read_text(encoding="utf-8"))
    assert payload["model_type"] == "mdn_surrogate"
    assert payload["kernel_id"] == kernel.manifest.kernel_id
    assert payload["quality_gate"]["accepted"] is True
    assert payload["quality_gate"]["max_high_uncertainty_fraction"] == 0.01
    assert payload["quality_gate"]["default_max_high_uncertainty_fraction"] == 0.01
