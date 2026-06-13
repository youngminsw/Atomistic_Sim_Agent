from __future__ import annotations

from sim_agent.schemas._parse import JsonMap

from .types import LayerRenewalPlan, MDCampaignPlan, StrataRange


def md_campaign_plan_payload(campaign: MDCampaignPlan) -> JsonMap:
    return {
        "material_id": campaign.material_id,
        "ion_species": campaign.ion_species,
        "phases": list(campaign.phases),
        "protocol_id": campaign.protocol_id,
        "force_field_protocol_id": campaign.force_field_protocol_id,
        "physics_scope": campaign.physics_scope,
        "energy_strata": _strata_payload(campaign.energy_strata),
        "polar_strata": _strata_payload(campaign.polar_strata),
        "azimuth_strata": _strata_payload(campaign.azimuth_strata),
        "pre_state_descriptors": list(campaign.pre_state_descriptors),
        "layer_renewal": _layer_renewal_payload(campaign.layer_renewal),
        "event_probe_default": campaign.event_probe_default,
    }


def _strata_payload(strata: StrataRange) -> JsonMap:
    return {
        "axis": strata.axis,
        "unit": strata.unit,
        "minimum": strata.minimum,
        "maximum": strata.maximum,
        "bin_count": strata.bin_count,
    }


def _layer_renewal_payload(plan: LayerRenewalPlan) -> JsonMap:
    return {
        "removed_depth_threshold_nm": plan.removed_depth_threshold_nm,
        "renewal_action": plan.renewal_action,
        "residual_policy": plan.residual_policy,
    }
