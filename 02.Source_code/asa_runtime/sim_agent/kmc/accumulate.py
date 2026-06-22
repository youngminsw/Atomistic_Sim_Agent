from __future__ import annotations

from sim_agent.geometry import PatternGeometry3D
from sim_agent.ml_surrogate import SurrogateTrainingDataset, SurrogateTrainingRow

from .types import CellKey, EnergyDepositionCell, EnergyDepositionField, IonImpact, KMCTransportError


def accumulate_energy_deposition(
    geometry: PatternGeometry3D,
    dataset: SurrogateTrainingDataset,
    impacts: tuple[IonImpact, ...],
) -> EnergyDepositionField:
    rows = {row.event_id: row for row in dataset.rows}
    energy_by_key: dict[CellKey, float] = {}
    removal_by_key: dict[CellKey, float] = {}
    count_by_key: dict[CellKey, int] = {}
    sputter_by_key: dict[CellKey, float] = {}
    reflect_by_key: dict[CellKey, float] = {}
    material_by_key: dict[CellKey, str] = {}
    region_by_key: dict[CellKey, str] = {}
    law_by_key: dict[CellKey, str] = {}
    events_by_key: dict[CellKey, list[str]] = {}

    for impact in impacts:
        row = _row_for_impact(rows, impact)
        address = geometry.cell_at_nm(impact.x_nm, impact.y_nm, impact.z_nm)
        key = CellKey(address.ix, address.iy, address.iz)
        energy_by_key[key] = energy_by_key.get(key, 0.0) + row.targets.deposited_energy_eV
        removal_by_key[key] = removal_by_key.get(key, 0.0) + _removal_drive_nm(geometry, address.material_id, row)
        count_by_key[key] = count_by_key.get(key, 0) + 1
        sputter_by_key[key] = sputter_by_key.get(key, 0.0) + row.targets.sputter_probability
        reflect_by_key[key] = reflect_by_key.get(key, 0.0) + row.targets.reflection_probability
        material_by_key[key] = address.material_id
        region_by_key[key] = address.region
        law_by_key[key] = _removal_law(geometry, address.material_id)
        events_by_key.setdefault(key, []).append(impact.event_id)

    cells = tuple(
        _cell(
            key,
            energy_by_key,
            removal_by_key,
            count_by_key,
            sputter_by_key,
            reflect_by_key,
            material_by_key,
            region_by_key,
            law_by_key,
            events_by_key,
        )
        for key in sorted(energy_by_key, key=lambda item: (item.ix, item.iy, item.iz))
    )
    return EnergyDepositionField(
        feature_type=geometry.feature_type,
        geometry_manifest=geometry.export_manifest(),
        cells=cells,
    )


def _row_for_impact(rows: dict[str, SurrogateTrainingRow], impact: IonImpact) -> SurrogateTrainingRow:
    row = rows.get(impact.event_id)
    if row is None:
        raise KMCTransportError(f"surrogate_row_missing_for_impact:{impact.event_id}")
    return row


def _removal_drive_nm(geometry: PatternGeometry3D, material_id: str, row: SurrogateTrainingRow) -> float:
    if material_id == geometry.mask_material_id:
        if geometry.pr_selectivity <= 0.0:
            raise KMCTransportError("pr_selectivity_must_be_positive")
        return row.targets.removed_depth_nm / geometry.pr_selectivity
    return row.targets.removed_depth_nm


def _removal_law(geometry: PatternGeometry3D, material_id: str) -> str:
    if material_id == geometry.mask_material_id:
        return "mask_selectivity_scaled"
    return "target_surrogate_direct"


def _cell(
    key: CellKey,
    energy_by_key: dict[CellKey, float],
    removal_by_key: dict[CellKey, float],
    count_by_key: dict[CellKey, int],
    sputter_by_key: dict[CellKey, float],
    reflect_by_key: dict[CellKey, float],
    material_by_key: dict[CellKey, str],
    region_by_key: dict[CellKey, str],
    law_by_key: dict[CellKey, str],
    events_by_key: dict[CellKey, list[str]],
) -> EnergyDepositionCell:
    hit_count = count_by_key[key]
    return EnergyDepositionCell(
        key=key,
        material_id=material_by_key[key],
        region=region_by_key[key],
        hit_count=hit_count,
        deposited_energy_eV=energy_by_key[key],
        removal_drive_nm=removal_by_key[key],
        sputter_probability=sputter_by_key[key] / hit_count,
        reflection_probability=reflect_by_key[key] / hit_count,
        event_ids=tuple(events_by_key[key]),
        removal_law=law_by_key[key],
    )
