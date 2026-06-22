from __future__ import annotations

from typing import Final

from .types import CampaignPlanError, ExpertReference, LayerRenewalPlan, MDCampaignPlan, StrataRange


DEFAULT_PROTOCOL_ID: Final = "continuous_stratified_bombardment"
DEFAULT_FORCE_FIELD_PROTOCOL_ID: Final = "Si_Tersoff_ZBL_physical_v001"
DEFAULT_PHYSICS_SCOPE: Final = "physical_bombardment_v1"
DEFAULT_PRE_STATE_DESCRIPTORS: Final = (
    "amorphous_index",
    "damage_dose",
    "roughness_rms_nm",
    "rdf_order_features",
    "implanted_inert_fraction",
    "local_fluence",
    "removed_depth_nm",
)
ALLOWED_PHASES: Final = frozenset(("crystal", "amorphous"))


def plan_md_campaign(
    material_id: str,
    ion_species: str,
    phases: tuple[str, ...],
    energy_range_eV: tuple[float, float],
    polar_range_deg: tuple[float, float],
    azimuth_range_deg: tuple[float, float],
    extend_expert: ExpertReference | None = None,
    force_field_protocol_id: str = DEFAULT_FORCE_FIELD_PROTOCOL_ID,
    physics_scope: str = DEFAULT_PHYSICS_SCOPE,
    active_layer_thickness_nm: float = 1.0,
) -> MDCampaignPlan:
    _reject_cross_scope_extension(material_id, ion_species, force_field_protocol_id, physics_scope, extend_expert)
    _require_phases(phases)
    if active_layer_thickness_nm <= 0.0:
        raise CampaignPlanError("active_layer_thickness_must_be_positive")
    return MDCampaignPlan(
        material_id=material_id,
        ion_species=ion_species,
        phases=phases,
        protocol_id=DEFAULT_PROTOCOL_ID,
        force_field_protocol_id=force_field_protocol_id,
        physics_scope=physics_scope,
        energy_strata=_strata("energy_eV", "eV", energy_range_eV, 6),
        polar_strata=_strata("polar_deg", "deg", polar_range_deg, 6),
        azimuth_strata=_strata("azimuth_deg", "deg", azimuth_range_deg, 8),
        pre_state_descriptors=DEFAULT_PRE_STATE_DESCRIPTORS,
        layer_renewal=LayerRenewalPlan(
            removed_depth_threshold_nm=active_layer_thickness_nm,
            renewal_action="expose_next_volume_state",
            residual_policy="reset_local_fluence_keep_configured_residuals",
        ),
        event_probe_default=False,
    )


def _reject_cross_scope_extension(
    material_id: str,
    ion_species: str,
    force_field_protocol_id: str,
    physics_scope: str,
    extend_expert: ExpertReference | None,
) -> None:
    if extend_expert is None:
        return
    same_scope = (
        extend_expert.material_id == material_id
        and extend_expert.ion_species == ion_species
        and extend_expert.force_field_protocol_id == force_field_protocol_id
        and extend_expert.physics_scope == physics_scope
    )
    if not same_scope:
        raise CampaignPlanError("new_campaign_required")


def _require_phases(phases: tuple[str, ...]) -> None:
    if not phases:
        raise CampaignPlanError("phase_required")
    for phase in phases:
        if phase not in ALLOWED_PHASES:
            raise CampaignPlanError("unsupported_phase")


def _strata(axis: str, unit: str, value_range: tuple[float, float], bin_count: int) -> StrataRange:
    minimum, maximum = value_range
    if minimum >= maximum:
        raise CampaignPlanError("invalid_strata_range")
    return StrataRange(axis=axis, unit=unit, minimum=minimum, maximum=maximum, bin_count=bin_count)
