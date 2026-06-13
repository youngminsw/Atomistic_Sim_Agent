from __future__ import annotations

from sim_agent.kmc import EnergyDepositionCell, EnergyDepositionField

from .types import LevelSetConfig, ProfileCellState, ProfileState, ProfileTimeline


def evolve_profile(field: EnergyDepositionField, config: LevelSetConfig) -> ProfileTimeline:
    states = tuple(_state(field, config, step_index) for step_index in range(config.time_steps + 1))
    return ProfileTimeline(feature_type=field.feature_type, states=states, cell_area_nm2=config.cell_area_nm2)


def _state(field: EnergyDepositionField, config: LevelSetConfig, step_index: int) -> ProfileState:
    fraction = step_index / config.time_steps
    cells = tuple(_cell(cell, fraction) for cell in field.cells)
    return ProfileState(
        step_index=step_index,
        time_s=step_index * config.time_step_s,
        cells=cells,
        total_removed_volume_nm3=sum(cell.surface_depth_nm * config.cell_area_nm2 for cell in cells),
    )


def _cell(cell: EnergyDepositionCell, fraction: float) -> ProfileCellState:
    return ProfileCellState(
        key=cell.key,
        material_id=cell.material_id,
        region=cell.region,
        surface_depth_nm=cell.removal_drive_nm * fraction,
        cumulative_energy_eV=cell.deposited_energy_eV * fraction,
        removal_law=cell.removal_law,
        event_ids=cell.event_ids,
    )
