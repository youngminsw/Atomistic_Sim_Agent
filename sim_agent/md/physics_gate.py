from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_bool, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .box_gate import MDBoxReadinessReport, assess_md_box_readiness
from .lammps_contract import DEFAULT_OUTPUT_SPECS


PRODUCTION_MIN_INCIDENTS: Final = 500


@dataclass(frozen=True, slots=True)
class MDPhysicsReadinessReport:
    ok: bool
    production_ready: bool
    payload: JsonMap


def assess_md_physics_readiness(
    manifest_payload: JsonMap,
    contract_payload: JsonMap,
    incident_schedule_payload: JsonMap,
    surface_state_payload: JsonMap,
    production_min_incidents: int = PRODUCTION_MIN_INCIDENTS,
) -> MDPhysicsReadinessReport:
    errors: list[str] = []
    blockers: list[str] = []
    evidence: list[str] = []

    _record_units(contract_payload, errors, evidence)
    _record_force_field(contract_payload, errors, evidence)
    _record_zbl_treatment(contract_payload, errors, evidence)
    _record_required_outputs(contract_payload, errors, evidence)
    incident_count = _record_incident_schedule(
        incident_schedule_payload,
        production_min_incidents,
        errors,
        blockers,
        evidence,
    )
    box_report = assess_md_box_readiness(surface_state_payload.get("md_box"))
    _record_surface_state(surface_state_payload, box_report, blockers, evidence)
    _record_manifest_identity(manifest_payload, errors, evidence)

    ok = not errors
    production_ready = ok and not blockers
    gate_status = _gate_status(ok, production_ready)
    return MDPhysicsReadinessReport(
        ok=ok,
        production_ready=production_ready,
        payload={
            "ok": ok,
            "production_ready": production_ready,
            "gate_status": gate_status,
            "incident_count": incident_count,
            "production_min_incidents": production_min_incidents,
            "md_box": box_report.payload,
            "evidence": evidence,
            "blockers": blockers,
            "errors": errors,
        },
    )


def _record_units(contract_payload: JsonMap, errors: list[str], evidence: list[str]) -> None:
    unit_style = _optional_text(contract_payload, "unit_style")
    energy_unit = _optional_text(contract_payload, "energy_unit")
    if unit_style == "metal" and energy_unit == "eV":
        evidence.append("unit_energy_contract_present")
        return
    errors.append("unit_energy_contract_invalid")


def _record_force_field(contract_payload: JsonMap, errors: list[str], evidence: list[str]) -> None:
    source_url = _optional_text(contract_payload, "force_field_source_url")
    protocol_id = _optional_text(contract_payload, "force_field_protocol_id")
    if source_url and protocol_id:
        evidence.append("force_field_provenance_present")
        return
    errors.append("force_field_provenance_missing")


def _record_zbl_treatment(
    contract_payload: JsonMap,
    errors: list[str],
    evidence: list[str],
) -> None:
    try:
        zbl_required = as_bool(require(contract_payload, "zbl_required"), "zbl_required")
    except SchemaValidationError:
        errors.append("zbl_requirement_missing")
        return
    model = _optional_text(contract_payload, "high_energy_collision_model")
    if zbl_required and "zbl" in model:
        evidence.append("zbl_collision_treatment_present")
        return
    if zbl_required:
        errors.append("zbl_collision_treatment_missing")
        return
    evidence.append("zbl_collision_treatment_not_required")


def _record_required_outputs(
    contract_payload: JsonMap,
    errors: list[str],
    evidence: list[str],
) -> None:
    required_outputs = _text_tuple(contract_payload, "required_outputs", errors)
    canonical = tuple(spec.filename for spec in DEFAULT_OUTPUT_SPECS)
    missing = tuple(filename for filename in canonical if filename not in required_outputs)
    if missing:
        errors.append("required_outputs_incomplete")
        return
    evidence.append("required_outputs_complete")


def _record_incident_schedule(
    incident_schedule_payload: JsonMap,
    production_min_incidents: int,
    errors: list[str],
    blockers: list[str],
    evidence: list[str],
) -> int:
    incident_count = _positive_int(incident_schedule_payload, "incident_count", errors)
    events = _sequence_length(incident_schedule_payload, "events", errors)
    if incident_count > 0:
        evidence.append("incident_schedule_present")
    if incident_count > 0 and events >= 0 and events != incident_count:
        blockers.append("incident_schedule_count_mismatch")
    if 0 < incident_count < production_min_incidents:
        blockers.append(
            f"production_incident_count_too_low:{incident_count}<{production_min_incidents}"
        )
    return incident_count


def _record_surface_state(
    surface_state_payload: JsonMap,
    box_report: MDBoxReadinessReport,
    blockers: list[str],
    evidence: list[str],
) -> None:
    blockers.extend(box_report.blockers)
    evidence.extend(box_report.evidence)
    if "descriptor_values" in surface_state_payload:
        evidence.append("surface_descriptor_values_present")
    else:
        blockers.append("surface_descriptor_values_missing")


def _record_manifest_identity(
    manifest_payload: JsonMap,
    errors: list[str],
    evidence: list[str],
) -> None:
    if _optional_text(manifest_payload, "material_id") and _optional_text(
        manifest_payload,
        "ion_species",
    ):
        evidence.append("manifest_material_ion_identity_present")
        return
    errors.append("manifest_material_ion_identity_missing")


def _gate_status(ok: bool, production_ready: bool) -> str:
    if production_ready:
        return "production_ready"
    if ok:
        return "smoke_only"
    return "physics_rejected"


def _text_tuple(payload: JsonMap, field: str, errors: list[str]) -> tuple[str, ...]:
    try:
        values = as_sequence(require(payload, field), field)
    except SchemaValidationError:
        errors.append(f"{field}_missing")
        return ()
    parsed: list[str] = []
    for index, value in enumerate(values):
        try:
            parsed.append(as_str(value, f"{field}[{index}]"))
        except SchemaValidationError:
            errors.append(f"{field}_invalid")
    return tuple(parsed)


def _positive_int(payload: JsonMap, field: str, errors: list[str]) -> int:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    errors.append(f"{field}_invalid")
    return 0


def _sequence_length(payload: JsonMap, field: str, errors: list[str]) -> int:
    try:
        return len(as_sequence(require(payload, field), field))
    except SchemaValidationError:
        errors.append(f"{field}_missing")
        return -1


def _optional_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return ""
