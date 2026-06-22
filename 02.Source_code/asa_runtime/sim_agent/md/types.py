from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MDRunStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


class MDVerificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedMDEvent:
    event_id: str
    ion: str
    material_id: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float
    amorphous_index: float
    roughness_rms_nm: float
    prior_removed_depth_nm: float
    damage_dose: float
    implanted_inert_fraction: float
    local_fluence: float
    rdf_crystal_similarity: float
    rdf_amorphous_similarity: float
    event_type: str
    yield_atoms_per_ion: float
    reflected: bool
    reflection_energy_out_eV: float
    reflection_polar_deg: float
    reflection_azimuth_deg: float
    implant_retained_fraction: float
    implant_depth_mean_nm: float
    deposited_energy_eV: float
    removed_depth_nm: float


@dataclass(frozen=True, slots=True)
class MDEventDataset:
    path: Path
    events: tuple[ParsedMDEvent, ...]
    event_count: int
    total_deposited_energy_eV: float
    total_removed_depth_nm: float
    reflected_count: int
    sputtered_count: int


@dataclass(frozen=True, slots=True)
class LammpsLogCheck:
    path: Path
    evidence: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventDatasetCheck:
    dataset: MDEventDataset | None
    evidence: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MDVerificationReport:
    ok: bool
    status: MDRunStatus
    dataset: MDEventDataset | None
    evidence: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedMDRunReport:
    ok: bool
    event_count: int
    descriptors_present: bool
    layer_removed_count: int
    total_deposited_energy_eV: float
    output_path: Path | None
    evidence: tuple[str, ...]
    errors: tuple[str, ...]
