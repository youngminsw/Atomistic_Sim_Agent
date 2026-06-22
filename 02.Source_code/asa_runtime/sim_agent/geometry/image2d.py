from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .image_mask import BinaryMask2D, PixelIndex, load_png_mask
from .types import GeometryError


@dataclass(frozen=True, slots=True)
class Normal2D:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class MaterialCell2D:
    ix: int
    iy: int
    x_nm: float
    y_nm: float
    material_id: str
    region: str
    is_opening: bool
    signed_distance_nm: float
    normal: Normal2D


@dataclass(frozen=True, slots=True)
class PatternGeometry2D:
    source_path: str
    width_px: int
    height_px: int
    pixel_size_nm: float
    target_material_id: str
    mask_material_id: str
    structure_description: str
    material_ids: tuple[str, ...]
    signed_distance_nm: tuple[float, ...]
    normals: tuple[Normal2D, ...]

    @property
    def width_nm(self) -> float:
        return self.width_px * self.pixel_size_nm

    @property
    def height_nm(self) -> float:
        return self.height_px * self.pixel_size_nm

    def cell_at_nm(self, x_nm: float, y_nm: float) -> MaterialCell2D:
        if x_nm < 0.0 or x_nm >= self.width_nm or y_nm < 0.0 or y_nm >= self.height_nm:
            raise GeometryError("click_outside_2d_geometry_bounds")
        ix = min(int(x_nm / self.pixel_size_nm), self.width_px - 1)
        iy = min(int(y_nm / self.pixel_size_nm), self.height_px - 1)
        return self.cell_at_pixel(ix, iy)

    def cell_at_pixel(self, ix: int, iy: int) -> MaterialCell2D:
        if ix < 0 or ix >= self.width_px or iy < 0 or iy >= self.height_px:
            raise GeometryError("pixel_outside_2d_geometry_bounds")
        index = iy * self.width_px + ix
        material_id = self.material_ids[index]
        is_opening = material_id == self.target_material_id
        return MaterialCell2D(
            ix=ix,
            iy=iy,
            x_nm=(ix + 0.5) * self.pixel_size_nm,
            y_nm=(iy + 0.5) * self.pixel_size_nm,
            material_id=material_id,
            region="opening" if is_opening else "mask",
            is_opening=is_opening,
            signed_distance_nm=self.signed_distance_nm[index],
            normal=self.normals[index],
        )

    def normals_valid(self) -> bool:
        return all(abs((normal.x * normal.x + normal.y * normal.y) - 1.0) < 1.0e-6 for normal in self.normals)

    def write_preview_ppm(self, path: Path) -> Path:
        lines = ["P3", f"{self.width_px} {self.height_px}", "255"]
        for y in range(self.height_px):
            row: list[str] = []
            for x in range(self.width_px):
                cell = self.cell_at_pixel(x, y)
                row.append("42 110 92" if cell.is_opening else "210 77 47")
            lines.append(" ".join(row))
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path


def load_pattern_geometry_2d(
    image_path: Path,
    pixel_size_nm: float,
    target_material_id: str,
    mask_material_id: str,
    structure_description: str,
) -> PatternGeometry2D:
    if pixel_size_nm <= 0.0:
        raise GeometryError("pixel_size_required")
    mask = load_png_mask(image_path, target_material_id=target_material_id, mask_material_id=mask_material_id)
    signed_distance = _signed_distance(mask, pixel_size_nm)
    return PatternGeometry2D(
        source_path=str(image_path),
        width_px=mask.width_px,
        height_px=mask.height_px,
        pixel_size_nm=pixel_size_nm,
        target_material_id=target_material_id,
        mask_material_id=mask_material_id,
        structure_description=structure_description,
        material_ids=tuple(
            mask.target_material_id if PixelIndex(index % mask.width_px, index // mask.width_px) in mask.opening_pixels else mask.mask_material_id
            for index in range(mask.width_px * mask.height_px)
        ),
        signed_distance_nm=signed_distance,
        normals=_normals(signed_distance, mask.width_px, mask.height_px),
    )


def _signed_distance(mask: BinaryMask2D, pixel_size_nm: float) -> tuple[float, ...]:
    values: list[float] = []
    for y in range(mask.height_px):
        for x in range(mask.width_px):
            point = PixelIndex(x, y)
            inside = point in mask.opening_pixels
            nearest = _nearest_opposite_distance_px(point, mask.opening_pixels, mask.width_px, mask.height_px, inside)
            sign = 1.0 if inside else -1.0
            values.append(sign * nearest * pixel_size_nm)
    return tuple(values)


def _nearest_opposite_distance_px(
    point: PixelIndex,
    opening_pixels: frozenset[PixelIndex],
    width_px: int,
    height_px: int,
    inside: bool,
) -> float:
    best = math.inf
    for y in range(height_px):
        for x in range(width_px):
            candidate = PixelIndex(x, y)
            candidate_inside = candidate in opening_pixels
            if candidate_inside == inside:
                continue
            distance = math.hypot(point.x - x, point.y - y)
            if distance < best:
                best = distance
    if math.isinf(best):
        return float(max(width_px, height_px))
    return max(best, 0.5)


def _normals(signed_distance: tuple[float, ...], width_px: int, height_px: int) -> tuple[Normal2D, ...]:
    normals: list[Normal2D] = []
    for y in range(height_px):
        for x in range(width_px):
            left = _sdf_at(signed_distance, width_px, height_px, x - 1, y)
            right = _sdf_at(signed_distance, width_px, height_px, x + 1, y)
            up = _sdf_at(signed_distance, width_px, height_px, x, y - 1)
            down = _sdf_at(signed_distance, width_px, height_px, x, y + 1)
            normals.append(_unit_normal(right - left, down - up))
    return tuple(normals)


def _sdf_at(values: tuple[float, ...], width_px: int, height_px: int, x: int, y: int) -> float:
    clamped_x = min(max(x, 0), width_px - 1)
    clamped_y = min(max(y, 0), height_px - 1)
    return values[clamped_y * width_px + clamped_x]


def _unit_normal(dx: float, dy: float) -> Normal2D:
    length = math.hypot(dx, dy)
    if length <= 1.0e-12:
        return Normal2D(0.0, 1.0)
    return Normal2D(dx / length, dy / length)
