from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .kernel import InteractionKernelManifest
from .types import SurrogateTargets, SurrogateTrainingDataset, SurrogateTrainingRow


EMPIRICAL_MDN_BACKEND: Final = "empirical_gaussian_mdn"
EMPIRICAL_MDN_ARTIFACT: Final = "empirical_mdn_model.json"
MIN_TARGET_SIGMA: Final = 1.0e-6


@dataclass(frozen=True, slots=True)
class EmpiricalMDNModel:
    artifact_path: Path
    payload: JsonMap


def write_empirical_mdn_model(
    output_dir: Path,
    manifest: InteractionKernelManifest,
    dataset: SurrogateTrainingDataset,
) -> EmpiricalMDNModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / EMPIRICAL_MDN_ARTIFACT
    payload = empirical_mdn_payload(manifest, dataset)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EmpiricalMDNModel(artifact_path=artifact_path, payload=payload)


def empirical_mdn_payload(
    manifest: InteractionKernelManifest,
    dataset: SurrogateTrainingDataset,
) -> JsonMap:
    sigma = _target_sigma(dataset.rows)
    return {
        "model_type": "mdn_surrogate",
        "training_backend": EMPIRICAL_MDN_BACKEND,
        "kernel_id": manifest.kernel_id,
        "material_id": manifest.material_id,
        "ion_species": manifest.ion_species,
        "force_field_protocol_id": manifest.force_field_protocol_id,
        "physics_scope": manifest.physics_scope,
        "training_event_count": dataset.row_count,
        "feature_columns": dataset.feature_columns,
        "output_columns": dataset.output_columns,
        "mixture_components": tuple(_component(row, dataset.row_count, sigma) for row in dataset.rows),
        "provenance_sources": manifest.provenance_sources,
    }


def _component(
    row: SurrogateTrainingRow,
    row_count: int,
    sigma: JsonMap,
) -> JsonMap:
    return {
        "component_id": row.event_id,
        "weight": 1.0 / row_count,
        "feature_centroid": row.feature_vector,
        "target_mean": _target_payload(row.targets),
        "target_sigma": sigma,
    }


def _target_payload(target: SurrogateTargets) -> JsonMap:
    return {
        "reflection_probability": target.reflection_probability,
        "sputter_probability": target.sputter_probability,
        "sputter_yield_atoms_per_ion": target.sputter_yield_atoms_per_ion,
        "reflection_energy_out_eV": target.reflection_energy_out_eV,
        "reflection_polar_deg": target.reflection_polar_deg,
        "reflection_azimuth_deg": target.reflection_azimuth_deg,
        "implant_retained_fraction": target.implant_retained_fraction,
        "implant_depth_mean_nm": target.implant_depth_mean_nm,
        "deposited_energy_eV": target.deposited_energy_eV,
        "removed_depth_nm": target.removed_depth_nm,
    }


def _target_sigma(rows: tuple[SurrogateTrainingRow, ...]) -> JsonMap:
    target_payloads = tuple(_target_payload(row.targets) for row in rows)
    return {
        field: _sigma(tuple(float(payload[field]) for payload in target_payloads))
        for field in target_payloads[0]
    }


def _sigma(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return MIN_TARGET_SIGMA
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(sqrt(variance), MIN_TARGET_SIGMA)
