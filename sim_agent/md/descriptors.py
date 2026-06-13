from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, float_field, float_map
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class PreStateDescriptor:
    amorphous_index: float
    damage_dose: float
    roughness_rms_nm: float
    removed_depth_nm: float
    rdf_order_features: dict[str, float]
    implanted_inert_fraction: float
    local_fluence: float


@dataclass(frozen=True, slots=True)
class DamageRecord:
    amorphous_index: float
    damage_dose: float
    roughness_rms_nm: float
    removed_depth_nm: float


@dataclass(frozen=True, slots=True)
class EnergyDepthRecord:
    depth_nm: float
    energy_eV: float


@dataclass(frozen=True, slots=True)
class RunDescriptorSet:
    active_layer_thickness_nm: float
    pre_states: dict[str, PreStateDescriptor]
    damage_records: dict[str, DamageRecord]
    energy_depth_records: dict[str, tuple[EnergyDepthRecord, ...]]


def load_run_descriptors(run_dir: Path) -> RunDescriptorSet:
    descriptor_payload = as_mapping(
        json.loads((run_dir / "roughness_rdf_descriptor.json").read_text(encoding="utf-8")),
        "roughness_rdf_descriptor",
    )
    return RunDescriptorSet(
        active_layer_thickness_nm=float_field(descriptor_payload, "active_layer_thickness_nm"),
        pre_states=_pre_states(as_mapping(descriptor_payload.get("events"), "events")),
        damage_records=load_damage_records(run_dir / "damage_profile.csv"),
        energy_depth_records=load_energy_depth_records(run_dir / "energy_depth_profile.csv"),
    )


def load_damage_records(path: Path) -> dict[str, DamageRecord]:
    records: dict[str, DamageRecord] = {}
    for row in _csv_rows(path):
        records[_field(row, "event_id")] = DamageRecord(
            amorphous_index=float(_field(row, "amorphous_index")),
            damage_dose=float(_field(row, "damage_dose")),
            roughness_rms_nm=float(_field(row, "roughness_rms_nm")),
            removed_depth_nm=float(_field(row, "removed_depth_nm")),
        )
    return records


def load_energy_depth_records(path: Path) -> dict[str, tuple[EnergyDepthRecord, ...]]:
    grouped: dict[str, list[EnergyDepthRecord]] = {}
    for row in _csv_rows(path):
        event_id = _field(row, "event_id")
        grouped.setdefault(event_id, []).append(
            EnergyDepthRecord(
                depth_nm=float(_field(row, "depth_nm")),
                energy_eV=float(_field(row, "energy_eV")),
            )
        )
    return {event_id: tuple(records) for event_id, records in grouped.items()}


def _pre_states(events: JsonMap) -> dict[str, PreStateDescriptor]:
    states: dict[str, PreStateDescriptor] = {}
    for event_id, raw_event in events.items():
        event = as_mapping(raw_event, f"events.{event_id}")
        state = as_mapping(event.get("pre_state"), f"events.{event_id}.pre_state")
        states[str(event_id)] = PreStateDescriptor(
            amorphous_index=float_field(state, "amorphous_index"),
            damage_dose=float_field(state, "damage_dose"),
            roughness_rms_nm=float_field(state, "roughness_rms_nm"),
            removed_depth_nm=float_field(state, "removed_depth_nm"),
            rdf_order_features=float_map(state.get("rdf_order_features", {}), "rdf_order_features"),
            implanted_inert_fraction=float_field(state, "implanted_inert_fraction"),
            local_fluence=float_field(state, "local_fluence"),
        )
    return states


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _field(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or not value:
        raise SchemaValidationError(f"{field} required")
    return value
