from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from sim_agent.schemas._parse import JsonMap, as_mapping, float_field, str_field
from sim_agent.schemas.errors import SchemaValidationError

from .descriptors import DamageRecord, EnergyDepthRecord, PreStateDescriptor, load_run_descriptors
from .dump_records import (
    ImplantRecord,
    IncidentRecord,
    ReflectionRecord,
    SputterRecord,
    load_implants,
    load_incidents,
    load_reflections,
    load_sputters,
)
from .types import ParsedMDRunReport
from .validation import validate_lammps_output_run


RecordT = TypeVar("RecordT")


def parse_lammps_output_run(
    run_dir: Path,
    material_id: str,
    descriptor_root: Path,
    out_path: Path,
) -> ParsedMDRunReport:
    gate_errors = validate_lammps_output_run(run_dir, material_id, descriptor_root)
    if gate_errors:
        return _failed_report(out_path, gate_errors)
    try:
        incidents = load_incidents(run_dir / "incident.dump")
        reflections = load_reflections(run_dir / "reflected.dump")
        sputters = load_sputters(run_dir / "sputtered.dump")
        implants = load_implants(run_dir / "implanted.dump")
        descriptors = load_run_descriptors(run_dir)
        payloads = tuple(
            _event_payload(
                incident,
                _record_for(reflections, incident.event_id, "reflection"),
                _record_for(sputters, incident.event_id, "sputter"),
                _record_for(implants, incident.event_id, "implant"),
                _record_for(descriptors.pre_states, incident.event_id, "pre_state"),
                _record_for(descriptors.damage_records, incident.event_id, "damage"),
                _record_for(descriptors.energy_depth_records, incident.event_id, "energy_depth"),
                descriptors.active_layer_thickness_nm,
            )
            for incident in incidents
        )
    except (FileNotFoundError, SchemaValidationError, ValueError, json.JSONDecodeError) as exc:
        return _failed_report(out_path, (f"md_parse_error:{exc}",))

    event_errors = tuple(error for payload in payloads for error in _event_errors(payload))
    if event_errors:
        return _failed_report(out_path, event_errors)
    _write_jsonl(out_path, payloads)
    return ParsedMDRunReport(
        ok=True,
        event_count=len(payloads),
        descriptors_present=bool(payloads),
        layer_removed_count=sum(1 for payload in payloads if payload["layer_removed"] is True),
        total_deposited_energy_eV=sum(float(payload["outcome"]["deposited_energy_eV"]) for payload in payloads),
        output_path=out_path,
        evidence=("md_events_parsed", "descriptors_present"),
        errors=(),
    )


def _event_payload(
    incident: IncidentRecord,
    reflection: ReflectionRecord,
    sputter: SputterRecord,
    implant: ImplantRecord,
    pre_state: PreStateDescriptor,
    damage: DamageRecord,
    energy_depth: tuple[EnergyDepthRecord, ...],
    active_layer_thickness_nm: float,
) -> JsonMap:
    deposited_energy = sum(record.energy_eV for record in energy_depth)
    layer_removed = pre_state.removed_depth_nm + damage.removed_depth_nm + 1e-12 >= active_layer_thickness_nm
    event_type = "reflect" if reflection.reflected else "sputter"
    return {
        "event_id": incident.event_id,
        "ion": incident.ion,
        "material_id": incident.material_id,
        "energy_eV": incident.energy_eV,
        "polar_deg": incident.polar_deg,
        "azimuth_deg": incident.azimuth_deg,
        "surface_state": _surface_state(pre_state),
        "pre_state": _pre_state(pre_state),
        "post_delta": _post_delta(damage),
        "outcome": {
            "event_type": event_type,
            "yield_atoms_per_ion": sputter.yield_atoms_per_ion,
            "reflected": reflection.reflected,
            "deposited_energy_eV": deposited_energy,
            "removed_depth_nm": damage.removed_depth_nm,
        },
        "reflection": {
            "energy_out_eV": reflection.energy_out_eV,
            "polar_deg": reflection.polar_deg,
            "azimuth_deg": reflection.azimuth_deg,
        },
        "implantation": {
            "retained_fraction": implant.retained_fraction,
            "depth_mean_nm": implant.depth_mean_nm,
        },
        "energy_depth_profile": tuple(
            {"depth_nm": record.depth_nm, "energy_eV": record.energy_eV} for record in energy_depth
        ),
        "layer_removed": layer_removed,
    }


def _surface_state(pre_state: PreStateDescriptor) -> JsonMap:
    return {
        "amorphous_index": pre_state.amorphous_index,
        "roughness_rms_nm": pre_state.roughness_rms_nm,
        "removed_depth_nm": pre_state.removed_depth_nm,
    }


def _pre_state(pre_state: PreStateDescriptor) -> JsonMap:
    return {
        "amorphous_index": pre_state.amorphous_index,
        "damage_dose": pre_state.damage_dose,
        "roughness_rms_nm": pre_state.roughness_rms_nm,
        "removed_depth_nm": pre_state.removed_depth_nm,
        "rdf_order_features": pre_state.rdf_order_features,
        "implanted_inert_fraction": pre_state.implanted_inert_fraction,
        "local_fluence": pre_state.local_fluence,
    }


def _post_delta(damage: DamageRecord) -> JsonMap:
    return {
        "amorphous_index": damage.amorphous_index,
        "damage_dose": damage.damage_dose,
        "roughness_rms_nm": damage.roughness_rms_nm,
        "removed_depth_nm": damage.removed_depth_nm,
    }


def _event_errors(payload: JsonMap) -> tuple[str, ...]:
    event_id = str_field(payload, "event_id")
    energy = float_field(payload, "energy_eV")
    outcome = as_mapping(payload.get("outcome"), "outcome")
    reflection = as_mapping(payload.get("reflection"), "reflection")
    implantation = as_mapping(payload.get("implantation"), "implantation")
    errors: list[str] = []
    if energy <= 0.0:
        errors.append(f"incident_energy_not_positive:{event_id}")
    errors.extend(_bounded_errors(event_id, "incident_polar_deg", float_field(payload, "polar_deg"), 0.0, 90.0))
    errors.extend(
        _bounded_errors(event_id, "incident_azimuth_deg", float_field(payload, "azimuth_deg"), 0.0, 360.0)
    )
    deposited_energy = float_field(outcome, "deposited_energy_eV")
    if deposited_energy < 0.0:
        errors.append(f"deposited_energy_negative:{event_id}")
    if deposited_energy > energy:
        errors.append(f"deposited_energy_exceeds_incident_energy:{event_id}")
    if float_field(outcome, "removed_depth_nm") < 0.0:
        errors.append(f"removed_depth_negative:{event_id}")
    if float_field(outcome, "yield_atoms_per_ion") < 0.0:
        errors.append(f"sputter_yield_negative:{event_id}")
    reflected_energy = float_field(reflection, "energy_out_eV")
    if reflected_energy < 0.0:
        errors.append(f"reflected_energy_negative:{event_id}")
    if reflected_energy > energy:
        errors.append(f"reflected_energy_exceeds_incident_energy:{event_id}")
    errors.extend(
        _bounded_errors(event_id, "reflection_polar_deg", float_field(reflection, "polar_deg"), 0.0, 180.0)
    )
    errors.extend(
        _bounded_errors(event_id, "reflection_azimuth_deg", float_field(reflection, "azimuth_deg"), 0.0, 360.0)
    )
    retained_fraction = float_field(implantation, "retained_fraction")
    if retained_fraction < 0.0 or retained_fraction > 1.0:
        errors.append(f"implant_retained_fraction_out_of_range:{event_id}")
    if float_field(implantation, "depth_mean_nm") < 0.0:
        errors.append(f"implant_depth_negative:{event_id}")
    return tuple(errors)


def _bounded_errors(
    event_id: str,
    field_name: str,
    value: float,
    lower: float,
    upper: float,
) -> tuple[str, ...]:
    if value < lower or value > upper:
        return (f"{field_name}_out_of_range:{event_id}",)
    return ()


def _record_for(records: Mapping[str, RecordT], event_id: str, label: str) -> RecordT:
    try:
        return records[event_id]
    except KeyError as exc:
        raise SchemaValidationError(f"{label}_missing:{event_id}") from exc


def _write_jsonl(out_path: Path, payloads: tuple[JsonMap, ...]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads)
    out_path.write_text(text + "\n", encoding="utf-8")


def _failed_report(out_path: Path, errors: tuple[str, ...]) -> ParsedMDRunReport:
    return ParsedMDRunReport(
        ok=False,
        event_count=0,
        descriptors_present=False,
        layer_removed_count=0,
        total_deposited_energy_eV=0.0,
        output_path=out_path,
        evidence=(),
        errors=errors,
    )
