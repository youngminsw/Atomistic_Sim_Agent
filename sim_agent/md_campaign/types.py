from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CampaignPlanError(ValueError):
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class StrataRange:
    axis: str
    unit: str
    minimum: float
    maximum: float
    bin_count: int


@dataclass(frozen=True, slots=True)
class LayerRenewalPlan:
    removed_depth_threshold_nm: float
    renewal_action: str
    residual_policy: str


@dataclass(frozen=True, slots=True)
class ExpertReference:
    expert_id: str
    material_id: str
    ion_species: str
    force_field_protocol_id: str
    physics_scope: str


@dataclass(frozen=True, slots=True)
class MDCampaignPlan:
    material_id: str
    ion_species: str
    phases: tuple[str, ...]
    protocol_id: str
    force_field_protocol_id: str
    physics_scope: str
    energy_strata: StrataRange
    polar_strata: StrataRange
    azimuth_strata: StrataRange
    pre_state_descriptors: tuple[str, ...]
    layer_renewal: LayerRenewalPlan
    event_probe_default: bool
