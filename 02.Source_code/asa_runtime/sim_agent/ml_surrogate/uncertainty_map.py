from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .kernel import InteractionContext, InteractionKernelManifest, KernelInferenceReport


@dataclass(frozen=True, slots=True)
class UncertaintyMapSample:
    sample_id: str
    context: InteractionContext
    inference: KernelInferenceReport
    snapshot_id: str
    chemistry_requested: bool = False


def write_uncertainty_map(
    path: Path,
    run_id: str,
    manifest: InteractionKernelManifest,
    samples: tuple[UncertaintyMapSample, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(uncertainty_map_payload(run_id, manifest, samples), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def uncertainty_map_payload(
    run_id: str,
    manifest: InteractionKernelManifest,
    samples: tuple[UncertaintyMapSample, ...],
) -> JsonMap:
    return {
        "run_id": run_id,
        "expert": {
            "expert_id": manifest.kernel_id,
            "material_id": manifest.material_id,
            "ion_species": manifest.ion_species,
            "force_field_protocol_id": manifest.force_field_protocol_id,
            "physics_scope": manifest.physics_scope,
        },
        "samples": tuple(_sample_payload(sample) for sample in samples),
    }


def _sample_payload(sample: UncertaintyMapSample) -> JsonMap:
    context = sample.context
    return {
        "sample_id": sample.sample_id,
        "material_id": context.material_id,
        "ion_species": context.ion_species,
        "energy_eV": context.energy_eV,
        "polar_deg": context.polar_deg,
        "azimuth_deg": context.azimuth_deg,
        "uncertainty_score": sample.inference.bundle.uncertainty.score,
        "snapshot_id": sample.snapshot_id,
        "chemistry_requested": sample.chemistry_requested,
    }
