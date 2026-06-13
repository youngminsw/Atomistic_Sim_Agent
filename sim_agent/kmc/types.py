from __future__ import annotations

from dataclasses import dataclass

from sim_agent.geometry import FeatureType, GeometryManifest, GridShape, PatternGeometry3D


class KMCTransportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IonImpact:
    event_id: str
    x_nm: float
    y_nm: float
    z_nm: float
    time_step: int


@dataclass(frozen=True, slots=True)
class CellKey:
    ix: int
    iy: int
    iz: int


@dataclass(frozen=True, slots=True)
class EnergyDepositionCell:
    key: CellKey
    material_id: str
    region: str
    hit_count: int
    deposited_energy_eV: float
    removal_drive_nm: float
    sputter_probability: float
    reflection_probability: float
    event_ids: tuple[str, ...]
    removal_law: str


@dataclass(frozen=True, slots=True)
class LevelSetEnergySource:
    grid_shape: GridShape
    cells: tuple[EnergyDepositionCell, ...]
    total_removal_drive_nm: float


@dataclass(frozen=True, slots=True)
class EnergyDepositionField:
    feature_type: FeatureType
    geometry_manifest: GeometryManifest
    cells: tuple[EnergyDepositionCell, ...]

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def total_hit_count(self) -> int:
        return sum(cell.hit_count for cell in self.cells)

    @property
    def total_deposited_energy_eV(self) -> float:
        return sum(cell.deposited_energy_eV for cell in self.cells)

    @property
    def total_removal_drive_nm(self) -> float:
        return sum(cell.removal_drive_nm for cell in self.cells)

    def diagnostic_at_nm(self, geometry: PatternGeometry3D, x_nm: float, y_nm: float, z_nm: float) -> EnergyDepositionCell:
        address = geometry.cell_at_nm(x_nm, y_nm, z_nm)
        return self.cell_at_key(CellKey(address.ix, address.iy, address.iz))

    def cell_at_key(self, key: CellKey) -> EnergyDepositionCell:
        for cell in self.cells:
            if cell.key == key:
                return cell
        raise KMCTransportError("energy_cell_missing")

    def to_level_set_source(self) -> LevelSetEnergySource:
        return LevelSetEnergySource(
            grid_shape=self.geometry_manifest.grid_shape,
            cells=self.cells,
            total_removal_drive_nm=self.total_removal_drive_nm,
        )
