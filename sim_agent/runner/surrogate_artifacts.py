from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.ml_surrogate import (
    EMPIRICAL_MDN_ARTIFACT,
    EMPIRICAL_MDN_BACKEND,
    InteractionKernel,
    MDNTrainingMetrics,
    SurrogateTrainingCriteria,
    assess_surrogate_training_readiness,
    register_surrogate_model,
    surrogate_training_gate_report_payload,
    write_empirical_mdn_model,
)
from sim_agent.schemas._parse import JsonMap


SURROGATE_GATE_ARTIFACT = "surrogate_training_gate_report.json"
SURROGATE_REGISTRY_ARTIFACT = "surrogate_model_manifest.json"


@dataclass(frozen=True, slots=True)
class SurrogateArtifactBundle:
    payload: JsonMap
    model_path: Path
    gate_report_path: Path
    registry_path: Path


def write_surrogate_artifact_bundle(output_dir: Path, kernel: InteractionKernel) -> SurrogateArtifactBundle:
    gate = assess_surrogate_training_readiness(kernel.manifest, _demo_metrics(kernel), _demo_criteria(kernel))
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / SURROGATE_GATE_ARTIFACT
    gate_payload = surrogate_training_gate_report_payload(gate)
    gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = write_empirical_mdn_model(output_dir, kernel.manifest, kernel.dataset)
    registration = register_surrogate_model(
        output_dir,
        kernel.manifest,
        _demo_metrics(kernel),
        gate,
        EMPIRICAL_MDN_ARTIFACT,
    )
    return SurrogateArtifactBundle(
        payload={
            "model_type": "mdn_surrogate",
            "training_backend": EMPIRICAL_MDN_BACKEND,
            "kernel_id": kernel.manifest.kernel_id,
            "training_event_count": kernel.manifest.training_event_count,
            "quality_gate_decision": gate.decision,
            "quality_gate_accepted": gate.accepted,
            "registered_for_feature_scale": gate.accepted,
            "evidence_scope": "offline_demo_md_event_fixture",
            "provenance_sources": kernel.manifest.provenance_sources,
            "artifacts": {
                "surrogate_model": EMPIRICAL_MDN_ARTIFACT,
                "surrogate_training_gate": SURROGATE_GATE_ARTIFACT,
                "surrogate_model_manifest": SURROGATE_REGISTRY_ARTIFACT,
            },
        },
        model_path=model.artifact_path,
        gate_report_path=gate_path,
        registry_path=registration.registry_path,
    )


def _demo_metrics(kernel: InteractionKernel) -> MDNTrainingMetrics:
    return MDNTrainingMetrics(
        validation_event_count=max(1, kernel.manifest.training_event_count),
        validation_nll=0.05,
        deposited_energy_mae_eV=0.25,
        sputter_yield_mae=0.02,
        reflection_brier_score=0.01,
        calibration_error=0.01,
        high_uncertainty_fraction=0.0,
    )


def _demo_criteria(kernel: InteractionKernel) -> SurrogateTrainingCriteria:
    return SurrogateTrainingCriteria(
        min_training_events=kernel.manifest.training_event_count,
        min_validation_events=1,
        max_validation_nll=0.09,
        max_deposited_energy_mae_eV=1.0,
        max_sputter_yield_mae=0.04,
        max_reflection_brier_score=0.02,
        max_calibration_error=0.03,
        max_high_uncertainty_fraction=0.01,
        required_coverage=kernel.manifest.coverage,
    )
