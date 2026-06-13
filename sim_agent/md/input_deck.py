from __future__ import annotations

from sim_agent.schemas._parse import (
    JsonMap,
    as_bool,
    as_float,
    as_mapping,
    as_sequence,
    as_str,
    require,
)
from sim_agent.schemas.errors import SchemaValidationError

from .input_deck_types import (
    DeckContract,
    DeckSchedule,
    DeckSurface,
    IncidentSpec,
    LAMMPSInputDeck,
    LAMMPSInputDeckError,
)
from .input_script import render_lammps_script


def render_lammps_input_deck(
    contract_payload: JsonMap,
    schedule_payload: JsonMap,
    surface_state_payload: JsonMap,
) -> LAMMPSInputDeck:
    try:
        contract = _contract(contract_payload)
        schedule = _schedule(schedule_payload)
        surface = _surface(surface_state_payload)
    except SchemaValidationError as exc:
        raise LAMMPSInputDeckError(str(exc)) from exc
    _ensure_supported_pair(surface.material_id, schedule.ion_species)
    events = _events(schedule_payload)
    manifest = _manifest_payload(contract, schedule, surface, len(events))
    return LAMMPSInputDeck(
        input_script=render_lammps_script(contract, schedule, surface, events, manifest),
        manifest_payload=manifest,
    )


def _contract(payload: JsonMap) -> DeckContract:
    outputs = tuple(
        as_str(item, "required_outputs")
        for item in as_sequence(require(payload, "required_outputs"), "required_outputs")
    )
    return DeckContract(
        run_id=as_str(require(payload, "run_id"), "run_id"),
        unit_style=as_str(require(payload, "unit_style"), "unit_style"),
        required_outputs=outputs,
        zbl_required=as_bool(require(payload, "zbl_required"), "zbl_required"),
        high_energy_collision_model=as_str(
            require(payload, "high_energy_collision_model"),
            "high_energy_collision_model",
        ),
        force_field_protocol_id=as_str(
            require(payload, "force_field_protocol_id"),
            "force_field_protocol_id",
        ),
        force_field_source_url=as_str(
            require(payload, "force_field_source_url"),
            "force_field_source_url",
        ),
    )


def _schedule(payload: JsonMap) -> DeckSchedule:
    count_float = as_float(require(payload, "incident_count"), "incident_count")
    count = int(count_float)
    if float(count) != count_float or count <= 0:
        raise SchemaValidationError("incident_count must be a positive integer")
    return DeckSchedule(
        schedule_id=as_str(require(payload, "schedule_id"), "schedule_id"),
        sampling_policy=as_str(require(payload, "sampling_policy"), "sampling_policy"),
        incident_count=count,
        ion_species=as_str(require(payload, "ion_species"), "ion_species"),
    )


def _surface(payload: JsonMap) -> DeckSurface:
    return DeckSurface(
        surface_state_id=as_str(require(payload, "surface_state_id"), "surface_state_id"),
        material_id=as_str(require(payload, "material_id"), "material_id"),
        phase=as_str(require(payload, "phase"), "phase"),
    )


def _events(payload: JsonMap) -> tuple[IncidentSpec, ...]:
    raw_events = as_sequence(require(payload, "events"), "events")
    events = tuple(_event(as_mapping(item, "events[]")) for item in raw_events)
    schedule = _schedule(payload)
    if len(events) != schedule.incident_count:
        raise LAMMPSInputDeckError("incident_count_mismatch")
    return events


def _event(payload: JsonMap) -> IncidentSpec:
    return IncidentSpec(
        event_id=as_str(require(payload, "event_id"), "event_id"),
        ion_species=as_str(require(payload, "ion_species"), "ion_species"),
        energy_eV=as_float(require(payload, "energy_eV"), "energy_eV"),
        polar_deg=as_float(require(payload, "polar_deg"), "polar_deg"),
        azimuth_deg=as_float(require(payload, "azimuth_deg"), "azimuth_deg"),
    )


def _ensure_supported_pair(material_id: str, ion_species: str) -> None:
    if material_id == "Si" and ion_species == "Ar":
        return
    raise LAMMPSInputDeckError("lammps_deck_supports_ar_on_si_only")


def _manifest_payload(
    contract: DeckContract,
    schedule: DeckSchedule,
    surface: DeckSurface,
    event_count: int,
) -> JsonMap:
    return {
        "input_deck_id": f"{contract.run_id}-input-deck",
        "run_id": contract.run_id,
        "schedule_id": schedule.schedule_id,
        "surface_state_id": surface.surface_state_id,
        "incident_count": event_count,
        "ion_species": schedule.ion_species,
        "material_id": surface.material_id,
        "phase": surface.phase,
        "unit_style": contract.unit_style,
        "sampling_policy": schedule.sampling_policy,
        "random_sampling_used": False,
        "required_structure_input": "surface_snapshot_before.data",
        "required_outputs": list(contract.required_outputs),
        "zbl_required": contract.zbl_required,
        "high_energy_collision_model": contract.high_energy_collision_model,
        "force_field_protocol_id": contract.force_field_protocol_id,
        "force_field_source_url": contract.force_field_source_url,
    }
