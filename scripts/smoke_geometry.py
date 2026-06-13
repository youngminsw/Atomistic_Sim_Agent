from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.geometry import GeometryError, load_pattern_geometry_2d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("2d",))
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--pixel-size-nm", type=float)
    parser.add_argument("--target-material", default="Si")
    parser.add_argument("--mask-material", default="PR")
    parser.add_argument("--description", default="2D PR pattern image")
    parser.add_argument("--preview")
    parser.add_argument("--click-nm")
    args = parser.parse_args()

    try:
        if args.pixel_size_nm is None:
            raise GeometryError("pixel_size_required")
        geometry = load_pattern_geometry_2d(
            image_path=Path(args.fixture),
            pixel_size_nm=args.pixel_size_nm,
            target_material_id=args.target_material,
            mask_material_id=args.mask_material,
            structure_description=args.description,
        )
        preview_written = False
        if args.preview:
            geometry.write_preview_ppm(Path(args.preview))
            preview_written = True
        click = geometry.cell_at_nm(*_parse_click(args.click_nm)) if args.click_nm else None
    except GeometryError as exc:
        print(str(exc))
        return 1

    materials = ",".join(sorted({geometry.mask_material_id, geometry.target_material_id}))
    print("geometry_valid=true")
    print("mode=2d")
    print(f"width_px={geometry.width_px}")
    print(f"height_px={geometry.height_px}")
    print(f"pixel_size_nm={geometry.pixel_size_nm}")
    print(f"materials={materials}")
    print("sdf_written=true")
    print(f"normals_valid={str(geometry.normals_valid()).lower()}")
    print(f"preview_written={str(preview_written).lower()}")
    if click is not None:
        print(f"click_material={click.material_id}")
        print(f"click_region={click.region}")
        print(f"click_signed_distance_nm={click.signed_distance_nm:.6f}")
    return 0


def _parse_click(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise GeometryError("click_nm_must_be_x_y")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise GeometryError("click_nm_must_be_numeric") from exc


if __name__ == "__main__":
    raise SystemExit(main())
