from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.md_campaign import ActiveLearningPlan, ActiveLearningRequest, plan_active_learning_run
from sim_agent.ml_surrogate import InteractionKernelManifest
from sim_agent.schemas._parse import JsonMap
from sim_agent.transport import TransportHitRecord

from .artifacts import write_json


@dataclass(frozen=True, slots=True)
class ActiveLearningArtifactPaths:
    uncertainty_map: Path
    active_learning_plan: Path


def write_active_learning_artifacts(
    output_dir: Path,
    run_id: str,
    manifest: InteractionKernelManifest,
    hit_history: tuple[TransportHitRecord, ...],
) -> ActiveLearningArtifactPaths:
    paths = ActiveLearningArtifactPaths(
        uncertainty_map=output_dir / "uncertainty_map.json",
        active_learning_plan=output_dir / "active_learning_plan.json",
    )
    write_json(paths.uncertainty_map, _uncertainty_map_payload(run_id, manifest, hit_history))
    write_json(paths.active_learning_plan, _plan_payload(plan_active_learning_run(output_dir)))
    return paths


def _uncertainty_map_payload(
    run_id: str,
    manifest: InteractionKernelManifest,
    hit_history: tuple[TransportHitRecord, ...],
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
        "samples": [_sample_payload(hit) for hit in hit_history],
    }


def _sample_payload(hit: TransportHitRecord) -> JsonMap:
    return {
        "sample_id": f"hit-{hit.time_step:04d}",
        "material_id": hit.material_id,
        "ion_species": "Ar",
        "energy_eV": hit.energy_eV,
        "polar_deg": hit.polar_deg,
        "azimuth_deg": hit.azimuth_deg,
        "uncertainty_score": hit.uncertainty_score,
        "snapshot_id": f"snap-{hit.time_step:04d}",
        "chemistry_requested": False,
    }


def _plan_payload(plan: ActiveLearningPlan) -> JsonMap:
    return {
        "run_id": plan.run_id,
        "same_expert": plan.same_expert,
        "controlled_event_probe_allowed": plan.controlled_event_probe_allowed,
        "batch_size": plan.batch_size,
        "requests": [_request_payload(request) for request in plan.requests],
    }


def _request_payload(request: ActiveLearningRequest) -> JsonMap:
    return {
        "request_id": request.request_id,
        "protocol": request.protocol,
        "energy_range_eV": list(request.energy_range_eV),
        "polar_range_deg": list(request.polar_range_deg),
        "azimuth_range_deg": list(request.azimuth_range_deg),
        "sample_count": request.sample_count,
        "snapshot_reset_required": request.snapshot_reset_required,
        "source_snapshot_ids": list(request.source_snapshot_ids),
    }
