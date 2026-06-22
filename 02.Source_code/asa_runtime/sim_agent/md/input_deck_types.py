from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class LAMMPSInputDeckError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class IncidentSpec:
    event_id: str
    ion_species: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float

    @property
    def lammps_suffix(self) -> str:
        return self.event_id.replace("-", "_")


@dataclass(frozen=True, slots=True)
class LAMMPSInputDeck:
    input_script: str
    manifest_payload: JsonMap


@dataclass(frozen=True, slots=True)
class DeckContract:
    run_id: str
    unit_style: str
    required_outputs: tuple[str, ...]
    zbl_required: bool
    high_energy_collision_model: str
    force_field_protocol_id: str
    force_field_source_url: str


@dataclass(frozen=True, slots=True)
class DeckSchedule:
    schedule_id: str
    sampling_policy: str
    incident_count: int
    ion_species: str


@dataclass(frozen=True, slots=True)
class DeckSurface:
    surface_state_id: str
    material_id: str
    phase: str
