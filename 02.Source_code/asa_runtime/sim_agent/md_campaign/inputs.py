from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class MDCampaignInputError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


def incident_schedule_payload(campaign: JsonMap, request: JsonMap, incident_count: int) -> JsonMap:
    if incident_count <= 0:
        raise MDCampaignInputError("incident_count_must_be_positive")
    energy = as_mapping(require(campaign, "energy_strata"), "energy_strata")
    polar = as_mapping(require(campaign, "polar_strata"), "polar_strata")
    azimuth = as_mapping(require(campaign, "azimuth_strata"), "azimuth_strata")
    ion_species = as_str(require(campaign, "ion_species"), "ion_species")
    return {
        "schedule_id": f"{_request_id(request)}-incident-schedule",
        "sampling_policy": "deterministic_stratified_midpoint_v1",
        "incident_count": incident_count,
        "ion_species": ion_species,
        "events": [
            _incident_event(index, incident_count, ion_species, energy, polar, azimuth)
            for index in range(incident_count)
        ],
    }


def surface_state_payload(campaign: JsonMap, request: JsonMap) -> JsonMap:
    scene = as_mapping(require(request, "scene"), "scene")
    surface = as_mapping(require(scene, "surface_state"), "surface_state")
    layer = as_mapping(require(campaign, "layer_renewal"), "layer_renewal")
    payload: JsonMap = {
        "surface_state_id": f"{_request_id(request)}-surface-state",
        "material_id": as_str(require(surface, "material_id"), "material_id"),
        "phase": as_str(require(surface, "phase"), "phase"),
        "descriptor_fields": [
            as_str(item, "pre_state_descriptors")
            for item in as_sequence(
                require(campaign, "pre_state_descriptors"), "pre_state_descriptors"
            )
        ],
        "descriptor_values": _descriptor_values(surface),
        "layer_renewal_action": as_str(require(layer, "renewal_action"), "renewal_action"),
        "removed_depth_threshold_nm": as_float(
            require(layer, "removed_depth_threshold_nm"), "removed_depth_threshold_nm"
        ),
        "residual_policy": as_str(require(layer, "residual_policy"), "residual_policy"),
    }
    _copy_optional_mapping(surface, payload, "md_box")
    _copy_optional_mapping(surface, payload, "lammps_structure_source")
    return payload


def _incident_event(
    index: int,
    incident_count: int,
    ion_species: str,
    energy: JsonMap,
    polar: JsonMap,
    azimuth: JsonMap,
) -> JsonMap:
    return {
        "event_id": f"incident-{index + 1:06d}",
        "ion_species": ion_species,
        "energy_eV": _sample_midpoint(energy, index, incident_count),
        "polar_deg": _sample_midpoint(polar, index, incident_count),
        "azimuth_deg": _sample_midpoint(azimuth, index, incident_count),
    }


def _sample_midpoint(strata: JsonMap, index: int, count: int) -> float:
    minimum = as_float(require(strata, "minimum"), "minimum")
    maximum = as_float(require(strata, "maximum"), "maximum")
    return round(minimum + (maximum - minimum) * (index + 0.5) / count, 6)


def _descriptor_values(surface: JsonMap) -> JsonMap:
    return {
        "amorphous_index": as_float(require(surface, "amorphous_index"), "amorphous_index"),
        "damage_dose": as_float(require(surface, "damage_dose"), "damage_dose"),
        "roughness_rms_nm": as_float(require(surface, "roughness_rms_nm"), "roughness_rms_nm"),
        "implanted_inert_fraction": as_float(
            require(surface, "implanted_inert_fraction"), "implanted_inert_fraction"
        ),
        "local_fluence": as_float(require(surface, "local_fluence"), "local_fluence"),
        "removed_depth_nm": as_float(require(surface, "removed_depth_nm"), "removed_depth_nm"),
        "rdf_crystal_similarity": as_float(
            require(surface, "rdf_crystal_similarity"), "rdf_crystal_similarity"
        ),
        "rdf_amorphous_similarity": as_float(
            require(surface, "rdf_amorphous_similarity"), "rdf_amorphous_similarity"
        ),
        "rdf_order_features": {
            "crystal_similarity": as_float(
                require(surface, "rdf_crystal_similarity"), "rdf_crystal_similarity"
            ),
            "amorphous_similarity": as_float(
                require(surface, "rdf_amorphous_similarity"), "rdf_amorphous_similarity"
            ),
        },
    }


def _request_id(request: JsonMap) -> str:
    return as_str(require(request, "request_id"), "request_id")


def _copy_optional_mapping(source: JsonMap, target: JsonMap, field: str) -> None:
    value = source.get(field)
    if value is None:
        return
    try:
        target[field] = dict(as_mapping(value, field))
    except SchemaValidationError as exc:
        raise MDCampaignInputError(f"{field}_invalid") from exc
