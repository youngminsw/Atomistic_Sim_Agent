from __future__ import annotations

from sim_agent.geometry.types import parse_feature_type
from sim_agent.kmc import CellKey
from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.state import SurfaceState, VolumeState
from sim_agent.transport import TransportCell, TransportField

from .layer_renewal import renew_surface_state_if_needed
from .types import LevelSetConfig, ProfileCellState, ProfileState, ProfileTimeline


def evolve_transport_profile(
    field: TransportField,
    config: LevelSetConfig,
    process: JsonMap | None = None,
) -> ProfileTimeline:
    states = tuple(_state(field, config, step_index, None, "") for step_index in range(config.time_steps + 1))
    return ProfileTimeline(
        feature_type=parse_feature_type(field.feature_type),
        states=states,
        cell_area_nm2=config.cell_area_nm2,
        mode=field.mode,
        process=process,
    )


def evolve_transport_profile_with_layer_renewal(
    field: TransportField,
    config: LevelSetConfig,
    surface_state: SurfaceState,
    volume_states: tuple[VolumeState, ...],
    process: JsonMap | None = None,
) -> ProfileTimeline:
    max_removed_depth_nm = max((cell.removed_depth_nm for cell in field.cells), default=0.0)
    renewal = renew_surface_state_if_needed(surface_state, volume_states, max_removed_depth_nm)
    renewal_threshold_nm = None
    if renewal.renewed:
        renewal_threshold_nm = max(surface_state.active_layer_thickness_nm - surface_state.removed_depth_nm, 0.0)
    states = tuple(
        _state(
            field,
            config,
            step_index,
            renewal_threshold_nm,
            renewal.surface_state.phase if renewal.renewed else surface_state.phase,
        )
        for step_index in range(config.time_steps + 1)
    )
    return ProfileTimeline(
        feature_type=parse_feature_type(field.feature_type),
        states=states,
        cell_area_nm2=config.cell_area_nm2,
        mode=field.mode,
        process=process,
    )


def _state(
    field: TransportField,
    config: LevelSetConfig,
    step_index: int,
    renewal_threshold_nm: float | None,
    renewed_phase: str,
) -> ProfileState:
    fraction = step_index / config.time_steps
    cells = tuple(_cell(cell, fraction, renewal_threshold_nm, renewed_phase) for cell in field.cells)
    return ProfileState(
        step_index=step_index,
        time_s=step_index * config.time_step_s,
        cells=cells,
        total_removed_volume_nm3=sum(cell.surface_depth_nm * config.cell_area_nm2 for cell in cells),
    )


def _cell(
    cell: TransportCell,
    fraction: float,
    renewal_threshold_nm: float | None,
    renewed_phase: str,
) -> ProfileCellState:
    depth_nm = cell.removed_depth_nm * fraction
    layer_renewed = renewal_threshold_nm is not None and depth_nm >= renewal_threshold_nm
    return ProfileCellState(
        key=CellKey(cell.key.ix, cell.key.iy, cell.key.iz),
        material_id=cell.material_id,
        region=cell.region,
        surface_depth_nm=depth_nm,
        cumulative_energy_eV=cell.deposited_energy_eV * fraction,
        removal_law="transport_kernel_direct",
        event_ids=cell.event_ids,
        layer_renewed=layer_renewed,
        surface_phase=renewed_phase if layer_renewed else "",
    )
