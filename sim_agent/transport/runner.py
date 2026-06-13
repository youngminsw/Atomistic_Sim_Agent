from __future__ import annotations

from sim_agent.geometry import MaterialCell2D, PatternGeometry2D, PatternGeometry3D
from sim_agent.ml_surrogate import InteractionContext, InteractionKernelRegistry
from sim_agent.schemas.distributions import IonAngularDistribution, IonEnergyDistribution
from sim_agent.schemas.events import EventBundle

from .incidence import local_incidence_angle_deg
from .sampling import sample_ions
from .types import (
    IonSample,
    SurfaceNormal3D,
    TransportCell,
    TransportCellKey,
    TransportError,
    TransportField,
    TransportHitRecord,
    TransportResult,
)


def run_transport_3d(
    geometry: PatternGeometry3D,
    registry: InteractionKernelRegistry,
    energy_distribution: IonEnergyDistribution,
    angular_distribution: IonAngularDistribution,
    ion_count: int,
    seed: int,
    duration_s: float | None = None,
) -> TransportResult:
    samples = sample_ions(energy_distribution, angular_distribution, ion_count, seed, duration_s)
    hit_records: list[TransportHitRecord] = []
    accumulator = _FieldAccumulator(mode="3d", feature_type=geometry.feature_type.value)
    landing_points = _opening_points_3d(geometry, max(1, min(ion_count, 16)))
    for sample in samples:
        x_nm, y_nm, z_nm = landing_points[sample.time_step % len(landing_points)]
        address = geometry.cell_at_nm(x_nm, y_nm, z_nm)
        key = TransportCellKey(address.ix, address.iy, address.iz)
        incidence = local_incidence_angle_deg(sample.polar_deg, sample.azimuth_deg, SurfaceNormal3D(0.0, 0.0, -1.0))
        inference = registry.infer(
            _context(sample, address.material_id, incidence, accumulator.local_state(key))
        )
        bundle = inference.bundle
        hit_records.append(
            _hit_record(sample, x_nm, y_nm, z_nm, address.material_id, address.region, incidence, bundle)
        )
        accumulator.add_hit(key, address.material_id, address.region, sample.event_id, bundle)
    field = accumulator.field()
    return TransportResult(mode="3d", feature_type=field.feature_type, field=field, hit_history=tuple(hit_records))


def run_transport_2d(
    geometry: PatternGeometry2D,
    registry: InteractionKernelRegistry,
    energy_distribution: IonEnergyDistribution,
    angular_distribution: IonAngularDistribution,
    ion_count: int,
    seed: int,
    duration_s: float | None = None,
) -> TransportResult:
    samples = sample_ions(energy_distribution, angular_distribution, ion_count, seed, duration_s)
    hit_records: list[TransportHitRecord] = []
    accumulator = _FieldAccumulator(mode="2d", feature_type="trench")
    landing_cells = _opening_cells_2d(geometry, max(1, min(ion_count, 16)))
    for sample in samples:
        cell = landing_cells[sample.time_step % len(landing_cells)]
        key = TransportCellKey(cell.ix, cell.iy, 0)
        normal = SurfaceNormal3D(cell.normal.x, 0.0, -cell.normal.y)
        incidence = local_incidence_angle_deg(sample.polar_deg, sample.azimuth_deg, normal)
        inference = registry.infer(
            _context(sample, cell.material_id, incidence, accumulator.local_state(key))
        )
        bundle = inference.bundle
        hit_records.append(
            _hit_record(sample, cell.x_nm, cell.y_nm, 0.0, cell.material_id, cell.region, incidence, bundle)
        )
        accumulator.add_hit(key, cell.material_id, cell.region, sample.event_id, bundle)
    field = accumulator.field()
    return TransportResult(mode="2d", feature_type=field.feature_type, field=field, hit_history=tuple(hit_records))


class _FieldAccumulator:
    def __init__(self, mode: str, feature_type: str) -> None:
        self._mode = mode
        self._feature_type = feature_type
        self._energy: dict[TransportCellKey, float] = {}
        self._removal: dict[TransportCellKey, float] = {}
        self._damage: dict[TransportCellKey, float] = {}
        self._roughness: dict[TransportCellKey, float] = {}
        self._implant: dict[TransportCellKey, float] = {}
        self._count: dict[TransportCellKey, int] = {}
        self._material: dict[TransportCellKey, str] = {}
        self._region: dict[TransportCellKey, str] = {}
        self._events: dict[TransportCellKey, list[str]] = {}

    def local_state(self, key: TransportCellKey) -> tuple[float, float, float, float, float]:
        count = self._count.get(key, 0)
        implant = 0.0 if count == 0 else self._implant.get(key, 0.0) / count
        return (
            self._damage.get(key, 0.0),
            max(self._roughness.get(key, 0.1), 0.1),
            implant,
            float(count),
            self._removal.get(key, 0.0),
        )

    def add_hit(
        self,
        key: TransportCellKey,
        material_id: str,
        region: str,
        event_id: str,
        bundle: EventBundle,
    ) -> None:
        self._energy[key] = self._energy.get(key, 0.0) + bundle.energy_transfer.deposited_energy_eV
        self._removal[key] = self._removal.get(key, 0.0) + bundle.removed_depth_nm
        self._damage[key] = self._damage.get(key, 0.0) + bundle.damage_delta.damage_dose
        self._roughness[key] = max(self._roughness.get(key, 0.1), bundle.damage_delta.roughness_rms_nm)
        self._implant[key] = self._implant.get(key, 0.0) + bundle.implantation.retained_fraction
        self._count[key] = self._count.get(key, 0) + 1
        self._material[key] = material_id
        self._region[key] = region
        self._events.setdefault(key, []).append(event_id)

    def field(self) -> TransportField:
        cells = tuple(self._cell(key) for key in sorted(self._count, key=lambda item: (item.ix, item.iy, item.iz)))
        return TransportField(mode=self._mode, feature_type=self._feature_type, cells=cells)

    def _cell(self, key: TransportCellKey) -> TransportCell:
        hit_count = self._count[key]
        return TransportCell(
            key=key,
            material_id=self._material[key],
            region=self._region[key],
            hit_count=hit_count,
            deposited_energy_eV=self._energy[key],
            removed_depth_nm=self._removal[key],
            damage_dose=self._damage[key],
            roughness_rms_nm=self._roughness[key],
            implanted_inert_fraction=self._implant[key] / hit_count,
            local_fluence=float(hit_count),
            event_ids=tuple(self._events[key]),
        )


def _context(
    sample: IonSample,
    material_id: str,
    local_incidence_deg: float,
    local_state: tuple[float, float, float, float, float],
) -> InteractionContext:
    damage_dose, roughness_rms_nm, implanted_fraction, local_fluence, removed_depth_nm = local_state
    return InteractionContext(
        ion_species="Ar",
        material_id=material_id,
        force_field_protocol_id="Si_Tersoff_ZBL_physical_v001",
        physics_scope="physical_bombardment_no_chemistry",
        energy_eV=sample.energy_eV,
        polar_deg=sample.polar_deg,
        azimuth_deg=sample.azimuth_deg,
        local_incidence_deg=local_incidence_deg,
        phase="crystal",
        amorphous_index=0.0,
        roughness_rms_nm=roughness_rms_nm,
        rdf_crystal_similarity=0.9,
        rdf_amorphous_similarity=0.1,
        damage_dose=damage_dose,
        implanted_inert_fraction=implanted_fraction,
        local_fluence=local_fluence,
        removed_depth_nm=removed_depth_nm,
    )


def _hit_record(
    sample: IonSample,
    x_nm: float,
    y_nm: float,
    z_nm: float,
    material_id: str,
    region: str,
    local_incidence_deg: float,
    bundle: EventBundle,
) -> TransportHitRecord:
    return TransportHitRecord(
        event_id=sample.event_id,
        time_step=sample.time_step,
        time_s=sample.time_s,
        x_nm=x_nm,
        y_nm=y_nm,
        z_nm=z_nm,
        material_id=material_id,
        region=region,
        energy_eV=sample.energy_eV,
        polar_deg=sample.polar_deg,
        azimuth_deg=sample.azimuth_deg,
        local_incidence_deg=local_incidence_deg,
        deposited_energy_eV=bundle.energy_transfer.deposited_energy_eV,
        removed_depth_nm=bundle.removed_depth_nm,
        uncertainty_ood=bundle.uncertainty.ood,
        uncertainty_score=bundle.uncertainty.score,
        uncertainty_reason=bundle.uncertainty.reason,
    )


def _opening_points_3d(geometry: PatternGeometry3D, limit: int) -> tuple[tuple[float, float, float], ...]:
    z_nm = geometry.bounds.z_min_nm + max(geometry.bounds.z_span_nm, 1.0) * 0.05
    candidates: list[tuple[float, float, float]] = []
    for iy in range(geometry.grid_shape.ny):
        for ix in range(geometry.grid_shape.nx):
            x_nm = geometry.bounds.x_min_nm + ((ix + 0.5) / geometry.grid_shape.nx) * geometry.bounds.x_span_nm
            y_nm = geometry.bounds.y_min_nm + ((iy + 0.5) / geometry.grid_shape.ny) * geometry.bounds.y_span_nm
            if geometry.cell_at_nm(x_nm, y_nm, z_nm).is_opening:
                candidates.append((x_nm, y_nm, z_nm))
    if not candidates:
        raise TransportError("opening_cell_required")
    return tuple(candidates[index] for index in _spread_indices(len(candidates), limit))


def _opening_cells_2d(geometry: PatternGeometry2D, limit: int) -> tuple[MaterialCell2D, ...]:
    candidates: list[MaterialCell2D] = []
    for y in range(geometry.height_px):
        for x in range(geometry.width_px):
            cell = geometry.cell_at_pixel(x, y)
            if cell.is_opening:
                candidates.append(cell)
    if not candidates:
        raise TransportError("opening_cell_required")
    return tuple(candidates[index] for index in _spread_indices(len(candidates), limit))


def _spread_indices(count: int, limit: int) -> tuple[int, ...]:
    capped = min(max(1, limit), count)
    if capped == 1:
        return (count // 2,)
    scale = (count - 1) / float(capped - 1)
    return tuple(int(round(index * scale)) for index in range(capped))
