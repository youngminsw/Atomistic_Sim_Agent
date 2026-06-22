from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .coverage import CoverageRange, KernelCoverage
from .kernel import InteractionKernelManifest
from .training_gate import (
    MDNTrainingMetrics,
    SurrogateTrainingGateReport,
    surrogate_training_gate_report_payload,
)


@dataclass(frozen=True, slots=True)
class SurrogateModelRegistryError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class SurrogateModelRegistration:
    registry_path: Path
    payload: JsonMap


def register_surrogate_model(
    output_dir: Path,
    manifest: InteractionKernelManifest,
    metrics: MDNTrainingMetrics,
    gate_report: SurrogateTrainingGateReport,
    model_artifact: str,
) -> SurrogateModelRegistration:
    if not gate_report.accepted:
        raise SurrogateModelRegistryError("surrogate_gate_not_accepted")
    payload = _registry_payload(manifest, metrics, gate_report, model_artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "surrogate_model_manifest.json"
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SurrogateModelRegistration(registry_path=registry_path, payload=payload)


def _registry_payload(
    manifest: InteractionKernelManifest,
    metrics: MDNTrainingMetrics,
    gate_report: SurrogateTrainingGateReport,
    model_artifact: str,
) -> JsonMap:
    return {
        "model_type": "mdn_surrogate",
        "kernel_id": manifest.kernel_id,
        "material_id": manifest.material_id,
        "ion_species": manifest.ion_species,
        "force_field_protocol_id": manifest.force_field_protocol_id,
        "physics_scope": manifest.physics_scope,
        "training_event_count": manifest.training_event_count,
        "model_artifact": model_artifact,
        "coverage": _coverage_payload(manifest.coverage),
        "metrics": {
            "validation_event_count": metrics.validation_event_count,
            "validation_nll": metrics.validation_nll,
            "deposited_energy_mae_eV": metrics.deposited_energy_mae_eV,
            "sputter_yield_mae": metrics.sputter_yield_mae,
            "reflection_brier_score": metrics.reflection_brier_score,
            "calibration_error": metrics.calibration_error,
            "high_uncertainty_fraction": metrics.high_uncertainty_fraction,
        },
        "quality_gate": surrogate_training_gate_report_payload(gate_report),
        "provenance_sources": manifest.provenance_sources,
    }


def _coverage_payload(coverage: KernelCoverage) -> JsonMap:
    return {
        "energy_eV": _range_payload(coverage.energy_eV),
        "polar_deg": _range_payload(coverage.polar_deg),
        "azimuth_deg": _range_payload(coverage.azimuth_deg),
    }


def _range_payload(value: CoverageRange) -> JsonMap:
    return {"minimum": value.minimum, "maximum": value.maximum}
