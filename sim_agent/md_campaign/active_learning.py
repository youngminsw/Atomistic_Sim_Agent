from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, as_sequence, float_field, str_field

from .types import ExpertReference


@dataclass(frozen=True, slots=True)
class ActiveLearningPlanError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class UncertaintySample:
    sample_id: str
    material_id: str
    ion_species: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float
    uncertainty_score: float
    snapshot_id: str
    chemistry_requested: bool


@dataclass(frozen=True, slots=True)
class ActiveLearningRequest:
    request_id: str
    protocol: str
    handoff_target: str
    force_field_protocol_id: str
    physics_scope: str
    energy_range_eV: tuple[float, float]
    polar_range_deg: tuple[float, float]
    azimuth_range_deg: tuple[float, float]
    sample_count: int
    sample_ids: tuple[str, ...]
    uncertainty_threshold: float
    same_material_only: bool
    snapshot_reset_required: bool
    source_snapshot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActiveLearningPlan:
    run_id: str
    expert: ExpertReference
    same_expert: bool
    controlled_event_probe_allowed: bool
    requests: tuple[ActiveLearningRequest, ...]

    @property
    def batch_size(self) -> int:
        return len(self.requests)


def plan_active_learning_run(run_dir: Path, uncertainty_threshold: float = 0.5) -> ActiveLearningPlan:
    payload = _payload(run_dir / "uncertainty_map.json")
    run_id = str_field(payload, "run_id")
    expert = _expert(as_mapping(payload.get("expert"), "expert"))
    samples = tuple(_sample(as_mapping(item, "samples[]")) for item in as_sequence(payload.get("samples", ()), "samples"))
    _require_same_scope(expert, samples)
    high_uncertainty = tuple(sample for sample in samples if sample.uncertainty_score >= uncertainty_threshold)
    requests = () if not high_uncertainty else (
        _request(run_id, expert, high_uncertainty, uncertainty_threshold),
    )
    return ActiveLearningPlan(
        run_id=run_id,
        expert=expert,
        same_expert=True,
        controlled_event_probe_allowed=bool(requests),
        requests=requests,
    )


def _payload(path: Path) -> JsonMap:
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")), "uncertainty_map")
    except FileNotFoundError as exc:
        raise ActiveLearningPlanError("uncertainty_map_required") from exc
    except json.JSONDecodeError as exc:
        raise ActiveLearningPlanError("invalid_uncertainty_map") from exc


def _expert(value: JsonMap) -> ExpertReference:
    return ExpertReference(
        expert_id=str_field(value, "expert_id"),
        material_id=str_field(value, "material_id"),
        ion_species=str_field(value, "ion_species"),
        force_field_protocol_id=str_field(value, "force_field_protocol_id"),
        physics_scope=str_field(value, "physics_scope"),
    )


def _sample(value: JsonMap) -> UncertaintySample:
    return UncertaintySample(
        sample_id=str_field(value, "sample_id"),
        material_id=str_field(value, "material_id"),
        ion_species=str_field(value, "ion_species"),
        energy_eV=float_field(value, "energy_eV"),
        polar_deg=float_field(value, "polar_deg"),
        azimuth_deg=float_field(value, "azimuth_deg"),
        uncertainty_score=float_field(value, "uncertainty_score"),
        snapshot_id=str_field(value, "snapshot_id"),
        chemistry_requested=as_bool(value.get("chemistry_requested", False), "chemistry_requested"),
    )


def _require_same_scope(expert: ExpertReference, samples: tuple[UncertaintySample, ...]) -> None:
    for sample in samples:
        same_material = sample.material_id == expert.material_id and sample.ion_species == expert.ion_species
        if sample.chemistry_requested or not same_material:
            raise ActiveLearningPlanError("new_campaign_required")


def _request(
    run_id: str,
    expert: ExpertReference,
    samples: tuple[UncertaintySample, ...],
    uncertainty_threshold: float,
) -> ActiveLearningRequest:
    return ActiveLearningRequest(
        request_id=f"{run_id}-probe-001",
        protocol="controlled_event_probe",
        handoff_target="md_agent",
        force_field_protocol_id=expert.force_field_protocol_id,
        physics_scope=expert.physics_scope,
        energy_range_eV=_range(tuple(sample.energy_eV for sample in samples)),
        polar_range_deg=_range(tuple(sample.polar_deg for sample in samples)),
        azimuth_range_deg=_range(tuple(sample.azimuth_deg for sample in samples)),
        sample_count=len(samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
        uncertainty_threshold=uncertainty_threshold,
        same_material_only=True,
        snapshot_reset_required=True,
        source_snapshot_ids=tuple(sample.snapshot_id for sample in samples),
    )


def _range(values: tuple[float, ...]) -> tuple[float, float]:
    return min(values), max(values)
