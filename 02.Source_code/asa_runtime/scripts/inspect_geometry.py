from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.geometry import GeometryError, GridShape, load_pattern_geometry_from_scene, load_png_mask
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--click-nm", default="0,0,0")
    parser.add_argument("--mask")
    parser.add_argument("--pixel", default="0,0")
    args = parser.parse_args()

    try:
        if args.scene:
            return _inspect_scene(Path(args.scene), args.click_nm)
        if args.mask:
            return _inspect_mask(Path(args.mask), args.pixel)
    except (json.JSONDecodeError, OSError, GeometryError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print("geometry_input_required")
    return 1


def _inspect_scene(scene_path: Path, click_raw: str) -> int:
    scene = as_mapping(json.loads(scene_path.read_text(encoding="utf-8")), "scene")
    geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
    click_x, click_y, click_z = _float_triplet(click_raw)
    cell = geometry.cell_at_nm(click_x, click_y, click_z)
    print("geometry_ok=true")
    print(f"feature_type={geometry.feature_type.value}")
    print(f"bounds_x={geometry.bounds.x_min_nm},{geometry.bounds.x_max_nm}")
    print(f"click_material={cell.material_id}")
    print(f"click_region={cell.region}")
    print(f"click_index={cell.ix},{cell.iy},{cell.iz}")
    return 0


def _inspect_mask(mask_path: Path, pixel_raw: str) -> int:
    mask = load_png_mask(mask_path, target_material_id="Si", mask_material_id="PR")
    pixel_x, pixel_y = _int_pair(pixel_raw)
    pixel = mask.pixel_at(pixel_x, pixel_y)
    print("mask_ok=true")
    print(f"mask_size={mask.width_px}x{mask.height_px}")
    print(f"opening_pixels={mask.opening_pixel_count}")
    print(f"pixel_material={pixel.material_id}")
    print(f"pixel_opening={str(pixel.is_opening).lower()}")
    return 0


def _float_triplet(raw: str) -> tuple[float, float, float]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise GeometryError("expected_three_nm_coordinates")
    return (_float(parts[0]), _float(parts[1]), _float(parts[2]))


def _int_pair(raw: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise GeometryError("expected_two_pixel_coordinates")
    return (_int(parts[0]), _int(parts[1]))


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise GeometryError(f"invalid_float={raw}") from exc


def _int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise GeometryError(f"invalid_int={raw}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
