from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, float_field, float_map, str_field

from .types import EventDatasetCheck, MDEventDataset, ParsedMDEvent


def inspect_md_events(
    path: Path,
    expected_events: int | None,
    required_ion: str | None,
    required_material: str | None,
) -> EventDatasetCheck:
    events: list[ParsedMDEvent] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            event = _parse_event(as_mapping(json.loads(raw_line), f"event_line_{line_number}"))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"event_parse_error:{line_number}:{exc}")
            continue
        errors.extend(_event_errors(event, required_ion, required_material))
        events.append(event)
    if expected_events is not None and len(events) != expected_events:
        errors.append(f"event_count_mismatch:expected={expected_events}:actual={len(events)}")
    if errors:
        return EventDatasetCheck(dataset=None, evidence=(), errors=tuple(errors))
    dataset = _dataset(path, tuple(events))
    return EventDatasetCheck(dataset=dataset, evidence=("md_events_sane",), errors=())


def _parse_event(mapping: JsonMap) -> ParsedMDEvent:
    outcome = as_mapping(mapping.get("outcome"), "outcome")
    state = _state_mapping(mapping)
    rdf = float_map(state.get("rdf_order_features", {}), "rdf_order_features")
    reflection = _optional_mapping(mapping.get("reflection"), "reflection")
    implantation = _optional_mapping(mapping.get("implantation"), "implantation")
    return ParsedMDEvent(
        event_id=str_field(mapping, "event_id"),
        ion=str_field(mapping, "ion"),
        material_id=str_field(mapping, "material_id"),
        energy_eV=float_field(mapping, "energy_eV"),
        polar_deg=float_field(mapping, "polar_deg"),
        azimuth_deg=float_field(mapping, "azimuth_deg"),
        amorphous_index=float_field(state, "amorphous_index"),
        roughness_rms_nm=float_field(state, "roughness_rms_nm"),
        prior_removed_depth_nm=float_field(state, "removed_depth_nm"),
        damage_dose=_float_or_default(state, "damage_dose"),
        implanted_inert_fraction=_float_or_default(state, "implanted_inert_fraction"),
        local_fluence=_float_or_default(state, "local_fluence"),
        rdf_crystal_similarity=rdf.get("crystal_similarity", 0.0),
        rdf_amorphous_similarity=rdf.get("amorphous_similarity", 0.0),
        event_type=str_field(outcome, "event_type"),
        yield_atoms_per_ion=float_field(outcome, "yield_atoms_per_ion"),
        reflected=as_bool(outcome.get("reflected"), "reflected"),
        reflection_energy_out_eV=_float_or_default(reflection, "energy_out_eV"),
        reflection_polar_deg=_float_or_default(reflection, "polar_deg"),
        reflection_azimuth_deg=_float_or_default(reflection, "azimuth_deg"),
        implant_retained_fraction=_float_or_default(implantation, "retained_fraction"),
        implant_depth_mean_nm=_float_or_default(implantation, "depth_mean_nm"),
        deposited_energy_eV=float_field(outcome, "deposited_energy_eV"),
        removed_depth_nm=float_field(outcome, "removed_depth_nm"),
    )


def _event_errors(event: ParsedMDEvent, required_ion: str | None, required_material: str | None) -> tuple[str, ...]:
    errors: list[str] = []
    if required_ion is not None and event.ion != required_ion:
        errors.append(f"unexpected_ion:{event.event_id}")
    if required_material is not None and event.material_id != required_material:
        errors.append(f"unexpected_material:{event.event_id}")
    if event.energy_eV <= 0.0:
        errors.append(f"incident_energy_not_positive:{event.event_id}")
    if event.deposited_energy_eV < 0.0:
        errors.append(f"deposited_energy_negative:{event.event_id}")
    if event.deposited_energy_eV > event.energy_eV:
        errors.append(f"deposited_energy_exceeds_incident_energy:{event.event_id}")
    if event.removed_depth_nm < 0.0:
        errors.append(f"removed_depth_negative:{event.event_id}")
    if event.yield_atoms_per_ion < 0.0:
        errors.append(f"sputter_yield_negative:{event.event_id}")
    if event.polar_deg < 0.0 or event.polar_deg > 90.0:
        errors.append(f"polar_angle_out_of_range:{event.event_id}")
    if event.azimuth_deg < 0.0 or event.azimuth_deg >= 360.0:
        errors.append(f"azimuth_angle_out_of_range:{event.event_id}")
    if event.damage_dose < 0.0:
        errors.append(f"damage_dose_negative:{event.event_id}")
    if event.implanted_inert_fraction < 0.0 or event.implanted_inert_fraction > 1.0:
        errors.append(f"implanted_inert_fraction_out_of_range:{event.event_id}")
    if event.local_fluence < 0.0:
        errors.append(f"local_fluence_negative:{event.event_id}")
    if event.rdf_crystal_similarity < 0.0 or event.rdf_crystal_similarity > 1.0:
        errors.append(f"rdf_crystal_similarity_out_of_range:{event.event_id}")
    if event.rdf_amorphous_similarity < 0.0 or event.rdf_amorphous_similarity > 1.0:
        errors.append(f"rdf_amorphous_similarity_out_of_range:{event.event_id}")
    if event.reflection_energy_out_eV < 0.0:
        errors.append(f"reflected_energy_negative:{event.event_id}")
    if event.reflection_energy_out_eV > event.energy_eV:
        errors.append(f"reflected_energy_exceeds_incident_energy:{event.event_id}")
    if event.reflection_polar_deg < 0.0 or event.reflection_polar_deg > 180.0:
        errors.append(f"reflection_polar_angle_out_of_range:{event.event_id}")
    if event.reflection_azimuth_deg < 0.0 or event.reflection_azimuth_deg >= 360.0:
        errors.append(f"reflection_azimuth_angle_out_of_range:{event.event_id}")
    if event.implant_retained_fraction < 0.0 or event.implant_retained_fraction > 1.0:
        errors.append(f"implant_retained_fraction_out_of_range:{event.event_id}")
    if event.implant_depth_mean_nm < 0.0:
        errors.append(f"implant_depth_negative:{event.event_id}")
    return tuple(errors)


def _state_mapping(mapping: JsonMap) -> JsonMap:
    pre_state = mapping.get("pre_state")
    if pre_state is not None:
        return as_mapping(pre_state, "pre_state")
    return as_mapping(mapping.get("surface_state"), "surface_state")


def _optional_mapping(value: object | None, label: str) -> JsonMap | None:
    if value is None:
        return None
    return as_mapping(value, label)


def _float_or_default(mapping: JsonMap | None, field_name: str) -> float:
    if mapping is None:
        return 0.0
    if mapping.get(field_name) is None:
        return 0.0
    return float_field(mapping, field_name)


def _dataset(path: Path, events: tuple[ParsedMDEvent, ...]) -> MDEventDataset:
    return MDEventDataset(
        path=path,
        events=events,
        event_count=len(events),
        total_deposited_energy_eV=sum(event.deposited_energy_eV for event in events),
        total_removed_depth_nm=sum(event.removed_depth_nm for event in events),
        reflected_count=sum(1 for event in events if event.reflected),
        sputtered_count=sum(1 for event in events if event.event_type == "sputter"),
    )
