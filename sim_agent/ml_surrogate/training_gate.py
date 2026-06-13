from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .coverage import CoverageRange, KernelCoverage
from .kernel import InteractionKernelManifest


DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION: Final = 0.01


@dataclass(frozen=True, slots=True)
class MDNTrainingMetrics:
    validation_event_count: int
    validation_nll: float
    deposited_energy_mae_eV: float
    sputter_yield_mae: float
    reflection_brier_score: float
    calibration_error: float
    high_uncertainty_fraction: float


@dataclass(frozen=True, slots=True)
class SurrogateTrainingCriteria:
    min_training_events: int
    min_validation_events: int
    max_validation_nll: float
    max_deposited_energy_mae_eV: float
    max_sputter_yield_mae: float
    max_reflection_brier_score: float
    max_calibration_error: float
    max_high_uncertainty_fraction: float
    required_coverage: KernelCoverage | None = None


@dataclass(frozen=True, slots=True)
class SurrogateTrainingGateReport:
    accepted: bool
    decision: str
    blockers: tuple[str, ...]
    evidence: tuple[str, ...]
    next_actions: tuple[str, ...]
    max_high_uncertainty_fraction: float


def assess_surrogate_training_readiness(
    manifest: InteractionKernelManifest,
    metrics: MDNTrainingMetrics,
    criteria: SurrogateTrainingCriteria,
) -> SurrogateTrainingGateReport:
    blockers: list[str] = []
    evidence: list[str] = []
    _record_data_sufficiency(manifest, metrics, criteria, blockers, evidence)
    _record_coverage(manifest.coverage, criteria.required_coverage, blockers, evidence)
    _record_validation_metrics(metrics, criteria, blockers, evidence)
    decision = _decision(blockers)
    return SurrogateTrainingGateReport(
        accepted=decision == "accepted_for_feature_scale",
        decision=decision,
        blockers=tuple(blockers),
        evidence=tuple(evidence),
        next_actions=_next_actions(decision),
        max_high_uncertainty_fraction=criteria.max_high_uncertainty_fraction,
    )


def surrogate_training_gate_report_payload(report: SurrogateTrainingGateReport) -> JsonMap:
    return {
        "accepted": report.accepted,
        "decision": report.decision,
        "blockers": report.blockers,
        "evidence": report.evidence,
        "next_actions": report.next_actions,
        "max_high_uncertainty_fraction": report.max_high_uncertainty_fraction,
        "default_max_high_uncertainty_fraction": DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION,
    }


def _record_data_sufficiency(
    manifest: InteractionKernelManifest,
    metrics: MDNTrainingMetrics,
    criteria: SurrogateTrainingCriteria,
    blockers: list[str],
    evidence: list[str],
) -> None:
    if manifest.training_event_count >= criteria.min_training_events:
        evidence.append("training_event_count_sufficient")
    else:
        blockers.append("training_event_count_too_low")
    if metrics.validation_event_count >= criteria.min_validation_events:
        evidence.append("validation_event_count_sufficient")
    else:
        blockers.append("validation_event_count_too_low")


def _record_coverage(
    coverage: KernelCoverage,
    required: KernelCoverage | None,
    blockers: list[str],
    evidence: list[str],
) -> None:
    if required is None:
        evidence.append("feature_space_coverage_not_required")
        return
    missing = (
        _coverage_gap("energy", coverage.energy_eV, required.energy_eV)
        + _coverage_gap("polar", coverage.polar_deg, required.polar_deg)
        + _coverage_gap("azimuth", coverage.azimuth_deg, required.azimuth_deg)
    )
    if missing:
        blockers.extend(missing)
    else:
        evidence.append("feature_space_coverage_sufficient")


def _record_validation_metrics(
    metrics: MDNTrainingMetrics,
    criteria: SurrogateTrainingCriteria,
    blockers: list[str],
    evidence: list[str],
) -> None:
    metric_blockers = (
        _max_blocker(
            metrics.validation_nll,
            criteria.max_validation_nll,
            "validation_nll_too_high",
        ),
        _max_blocker(
            metrics.deposited_energy_mae_eV,
            criteria.max_deposited_energy_mae_eV,
            "deposited_energy_mae_too_high",
        ),
        _max_blocker(
            metrics.sputter_yield_mae,
            criteria.max_sputter_yield_mae,
            "sputter_yield_mae_too_high",
        ),
        _max_blocker(
            metrics.reflection_brier_score,
            criteria.max_reflection_brier_score,
            "reflection_brier_score_too_high",
        ),
        _max_blocker(
            metrics.calibration_error,
            criteria.max_calibration_error,
            "calibration_error_too_high",
        ),
    )
    blockers.extend(blocker for blocker in metric_blockers if blocker)
    if metrics.high_uncertainty_fraction > criteria.max_high_uncertainty_fraction:
        blockers.append("high_uncertainty_fraction_too_high")
    if len(blockers) == 0:
        evidence.append("validation_metrics_within_thresholds")


def _coverage_gap(prefix: str, actual: CoverageRange, required: CoverageRange) -> tuple[str, ...]:
    blockers: list[str] = []
    if actual.minimum > required.minimum:
        blockers.append(f"{prefix}_coverage_min_missing")
    if actual.maximum < required.maximum:
        blockers.append(f"{prefix}_coverage_max_missing")
    return tuple(blockers)


def _max_blocker(value: float, threshold: float, blocker: str) -> str:
    if value > threshold:
        return blocker
    return ""


def _decision(blockers: list[str]) -> str:
    if not blockers:
        return "accepted_for_feature_scale"
    if any(_is_retrain_blocker(blocker) for blocker in blockers):
        return "retrain_required"
    return "active_learning_required"


def _is_retrain_blocker(blocker: str) -> bool:
    return blocker in (
        "validation_nll_too_high",
        "deposited_energy_mae_too_high",
        "sputter_yield_mae_too_high",
        "reflection_brier_score_too_high",
        "calibration_error_too_high",
    )


def _next_actions(decision: str) -> tuple[str, ...]:
    if decision == "accepted_for_feature_scale":
        return ("register_kernel_for_feature_scale",)
    if decision == "retrain_required":
        return ("retrain_mdn", "rerun_surrogate_training_gate")
    return ("plan_active_learning_md", "rerun_surrogate_training_gate")
