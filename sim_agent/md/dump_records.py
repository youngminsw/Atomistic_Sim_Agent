from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class IncidentRecord:
    event_id: str
    ion: str
    material_id: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    reflected: bool
    energy_out_eV: float
    polar_deg: float
    azimuth_deg: float


@dataclass(frozen=True, slots=True)
class SputterRecord:
    species: str
    yield_atoms_per_ion: float


@dataclass(frozen=True, slots=True)
class ImplantRecord:
    retained_fraction: float
    depth_mean_nm: float


def load_incidents(path: Path) -> tuple[IncidentRecord, ...]:
    return tuple(
        IncidentRecord(
            event_id=_field(row, "event_id"),
            ion=_field(row, "ion"),
            material_id=_field(row, "material_id"),
            energy_eV=float(_field(row, "energy_eV")),
            polar_deg=float(_field(row, "polar_deg")),
            azimuth_deg=float(_field(row, "azimuth_deg")),
        )
        for row in _csv_rows(path)
    )


def load_reflections(path: Path) -> dict[str, ReflectionRecord]:
    return {
        _field(row, "event_id"): ReflectionRecord(
            reflected=_bool(_field(row, "reflected")),
            energy_out_eV=float(_field(row, "energy_out_eV")),
            polar_deg=float(_field(row, "polar_deg")),
            azimuth_deg=float(_field(row, "azimuth_deg")),
        )
        for row in _csv_rows(path)
    }


def load_sputters(path: Path) -> dict[str, SputterRecord]:
    return {
        _field(row, "event_id"): SputterRecord(
            species=_field(row, "species"),
            yield_atoms_per_ion=float(_field(row, "yield_atoms_per_ion")),
        )
        for row in _csv_rows(path)
    }


def load_implants(path: Path) -> dict[str, ImplantRecord]:
    return {
        _field(row, "event_id"): ImplantRecord(
            retained_fraction=float(_field(row, "retained_fraction")),
            depth_mean_nm=float(_field(row, "depth_mean_nm")),
        )
        for row in _csv_rows(path)
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _field(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or not value:
        raise SchemaValidationError(f"{field} required")
    return value


def _bool(raw: str) -> bool:
    normalized = raw.lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise SchemaValidationError("boolean required")
