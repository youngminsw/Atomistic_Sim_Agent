from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class GeometryError(ValueError):
    pass


class FeatureType(StrEnum):
    TRENCH = "trench"
    HOLE = "hole"


@dataclass(frozen=True, slots=True)
class GridShape:
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise GeometryError("grid_shape_must_be_positive")


@dataclass(frozen=True, slots=True)
class Bounds3D:
    x_min_nm: float
    x_max_nm: float
    y_min_nm: float
    y_max_nm: float
    z_min_nm: float
    z_max_nm: float

    @property
    def x_span_nm(self) -> float:
        return self.x_max_nm - self.x_min_nm

    @property
    def y_span_nm(self) -> float:
        return self.y_max_nm - self.y_min_nm

    @property
    def z_span_nm(self) -> float:
        return self.z_max_nm - self.z_min_nm

    def with_min_depth(self, depth_nm: float) -> Bounds3D:
        if depth_nm <= 0.0:
            raise GeometryError("target_depth_nm_must_be_positive")
        if self.z_span_nm > 0.0:
            return self
        return Bounds3D(
            x_min_nm=self.x_min_nm,
            x_max_nm=self.x_max_nm,
            y_min_nm=self.y_min_nm,
            y_max_nm=self.y_max_nm,
            z_min_nm=self.z_min_nm,
            z_max_nm=self.z_min_nm + depth_nm,
        )

    def contains(self, x_nm: float, y_nm: float, z_nm: float) -> bool:
        return (
            self.x_min_nm <= x_nm <= self.x_max_nm
            and self.y_min_nm <= y_nm <= self.y_max_nm
            and self.z_min_nm <= z_nm <= self.z_max_nm
        )


@dataclass(frozen=True, slots=True)
class CellAddress:
    ix: int
    iy: int
    iz: int
    material_id: str
    region: str
    is_opening: bool


@dataclass(frozen=True, slots=True)
class GeometryManifest:
    feature_type: FeatureType
    grid_shape: GridShape
    bounds: Bounds3D
    source_path: str
    target_material_id: str
    mask_material_id: str
    pr_selectivity: float


@dataclass(frozen=True, slots=True)
class PatternGeometry3D:
    feature_type: FeatureType
    bounds: Bounds3D
    grid_shape: GridShape
    source_path: str
    target_material_id: str
    mask_material_id: str
    pr_selectivity: float

    def cell_at_nm(self, x_nm: float, y_nm: float, z_nm: float) -> CellAddress:
        if not self.bounds.contains(x_nm, y_nm, z_nm):
            raise GeometryError("click_outside_geometry_bounds")
        is_opening = self._is_opening(x_nm, y_nm)
        material_id = self.target_material_id if is_opening else self.mask_material_id
        region = "opening" if is_opening else "mask"
        return CellAddress(
            ix=_axis_index(x_nm, self.bounds.x_min_nm, self.bounds.x_max_nm, self.grid_shape.nx),
            iy=_axis_index(y_nm, self.bounds.y_min_nm, self.bounds.y_max_nm, self.grid_shape.ny),
            iz=_axis_index(z_nm, self.bounds.z_min_nm, self.bounds.z_max_nm, self.grid_shape.nz),
            material_id=material_id,
            region=region,
            is_opening=is_opening,
        )

    def export_manifest(self) -> GeometryManifest:
        return GeometryManifest(
            feature_type=self.feature_type,
            grid_shape=self.grid_shape,
            bounds=self.bounds,
            source_path=self.source_path,
            target_material_id=self.target_material_id,
            mask_material_id=self.mask_material_id,
            pr_selectivity=self.pr_selectivity,
        )

    def _is_opening(self, x_nm: float, y_nm: float) -> bool:
        x_center = self.bounds.x_min_nm + self.bounds.x_span_nm * 0.5
        y_center = self.bounds.y_min_nm + self.bounds.y_span_nm * 0.5
        match self.feature_type:
            case FeatureType.HOLE:
                radius_nm = min(self.bounds.x_span_nm, self.bounds.y_span_nm) * 0.25
                return (x_nm - x_center) ** 2 + (y_nm - y_center) ** 2 <= radius_nm**2
            case FeatureType.TRENCH:
                half_width_nm = self.bounds.x_span_nm * 0.125
                return abs(x_nm - x_center) <= half_width_nm
            case unreachable:
                assert_never(unreachable)


def parse_feature_type(value: str) -> FeatureType:
    try:
        return FeatureType(value)
    except ValueError as exc:
        raise GeometryError(f"unknown_feature_type={value}") from exc


def _axis_index(value: float, axis_min: float, axis_max: float, count: int) -> int:
    span = axis_max - axis_min
    if span <= 0.0:
        return 0
    scaled = int(((value - axis_min) / span) * count)
    return min(max(scaled, 0), count - 1)
