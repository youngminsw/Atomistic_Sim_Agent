from __future__ import annotations

from dataclasses import dataclass


class TransportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SurfaceNormal3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class IonSample:
    event_id: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float
    time_step: int
    time_s: float


@dataclass(frozen=True, slots=True)
class TransportCellKey:
    ix: int
    iy: int
    iz: int


@dataclass(frozen=True, slots=True)
class TransportCell:
    key: TransportCellKey
    material_id: str
    region: str
    hit_count: int
    deposited_energy_eV: float
    removed_depth_nm: float
    damage_dose: float
    roughness_rms_nm: float
    implanted_inert_fraction: float
    local_fluence: float
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransportHitRecord:
    event_id: str
    time_step: int
    time_s: float
    x_nm: float
    y_nm: float
    z_nm: float
    material_id: str
    region: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float
    local_incidence_deg: float
    deposited_energy_eV: float
    removed_depth_nm: float
    uncertainty_ood: bool
    uncertainty_score: float
    uncertainty_reason: str | None


@dataclass(frozen=True, slots=True)
class TransportField:
    mode: str
    feature_type: str
    cells: tuple[TransportCell, ...]

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
    def total_removed_depth_nm(self) -> float:
        return sum(cell.removed_depth_nm for cell in self.cells)


@dataclass(frozen=True, slots=True)
class TransportResult:
    mode: str
    feature_type: str
    field: TransportField
    hit_history: tuple[TransportHitRecord, ...]

    @property
    def hit_history_count(self) -> int:
        return len(self.hit_history)
