from __future__ import annotations

from dataclasses import dataclass

from sim_agent.geometry import FeatureType, PatternGeometry3D
from sim_agent.kmc import CellKey
from sim_agent.schemas._parse import JsonMap


class LevelSetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LevelSetConfig:
    time_steps: int
    time_step_s: float
    cell_area_nm2: float

    def __post_init__(self) -> None:
        if self.time_steps <= 0:
            raise LevelSetError("time_steps_must_be_positive")
        if self.time_step_s <= 0.0:
            raise LevelSetError("time_step_s_must_be_positive")
        if self.cell_area_nm2 <= 0.0:
            raise LevelSetError("cell_area_nm2_must_be_positive")


@dataclass(frozen=True, slots=True)
class ProfileCellState:
    key: CellKey
    material_id: str
    region: str
    surface_depth_nm: float
    cumulative_energy_eV: float
    removal_law: str
    event_ids: tuple[str, ...]
    layer_renewed: bool = False
    surface_phase: str = ""


@dataclass(frozen=True, slots=True)
class ProfileState:
    step_index: int
    time_s: float
    cells: tuple[ProfileCellState, ...]
    total_removed_volume_nm3: float

    def cell_at_key(self, key: CellKey) -> ProfileCellState:
        for cell in self.cells:
            if cell.key == key:
                return cell
        raise LevelSetError("profile_cell_missing")


@dataclass(frozen=True, slots=True)
class ProfileDiagnostic:
    key: CellKey
    material_id: str
    region: str
    depth_history_nm: tuple[float, ...]
    energy_history_eV: tuple[float, ...]
    removal_law: str
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileTimeline:
    feature_type: FeatureType
    states: tuple[ProfileState, ...]
    cell_area_nm2: float
    mode: str = "3d"
    process: JsonMap | None = None

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def final_state(self) -> ProfileState:
        if not self.states:
            raise LevelSetError("profile_timeline_empty")
        return self.states[-1]

    @property
    def renewal_count(self) -> int:
        return sum(1 for cell in self.final_state.cells if cell.layer_renewed)

    def diagnostic_at_nm(self, geometry: PatternGeometry3D, x_nm: float, y_nm: float, z_nm: float) -> ProfileDiagnostic:
        address = geometry.cell_at_nm(x_nm, y_nm, z_nm)
        key = CellKey(address.ix, address.iy, address.iz)
        cells = tuple(state.cell_at_key(key) for state in self.states)
        first = cells[0]
        return ProfileDiagnostic(
            key=key,
            material_id=first.material_id,
            region=first.region,
            depth_history_nm=tuple(cell.surface_depth_nm for cell in cells),
            energy_history_eV=tuple(cell.cumulative_energy_eV for cell in cells),
            removal_law=first.removal_law,
            event_ids=first.event_ids,
        )
