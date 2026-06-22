from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_scene(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "scenes" / name).read_text(encoding="utf-8"))


def test_hole_scene_geometry_maps_center_to_target_and_edge_to_pr() -> None:
    from sim_agent.geometry import GridShape, load_pattern_geometry_from_scene

    scene = _load_scene("pr_hole_scene.json")

    geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
    center = geometry.cell_at_nm(0.0, 0.0, 1.0)
    edge = geometry.cell_at_nm(18.0, 18.0, 1.0)

    assert geometry.feature_type.value == "hole"
    assert geometry.bounds.x_min_nm == -20.0
    assert geometry.bounds.x_max_nm == 20.0
    assert geometry.export_manifest().pr_selectivity == 20.0
    assert center.material_id == "Si"
    assert center.region == "opening"
    assert edge.material_id == "PR"
    assert edge.region == "mask"


def test_trench_geometry_uses_same_click_contract_as_hole() -> None:
    from sim_agent.geometry import GridShape, load_pattern_geometry_from_scene

    scene = {
        "mode": "3d",
        "feature_type": "trench",
        "geometry_source": {
            "kind": "mesh",
            "path": "tests/fixtures/geometry/pr_trench.stl",
            "units": "nm",
        },
        "target_material": "Si",
        "mask_material": "PR",
        "pr_selectivity": 30.0,
    }

    geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(40, 20, 8), target_depth_nm=16.0)
    center = geometry.cell_at_nm(20.0, 10.0, 0.5)
    side = geometry.cell_at_nm(2.0, 10.0, 0.5)

    assert geometry.feature_type.value == "trench"
    assert center.material_id == "Si"
    assert center.region == "opening"
    assert side.material_id == "PR"
    assert side.region == "mask"
    assert geometry.export_manifest().grid_shape == GridShape(40, 20, 8)


def test_png_mask_extraction_supports_2d_image_input() -> None:
    from sim_agent.geometry import load_png_mask

    mask = load_png_mask(FIXTURE_ROOT / "geometry" / "pr_hole_mask.png", target_material_id="Si", mask_material_id="PR")
    center = mask.pixel_at(8, 8)
    corner = mask.pixel_at(0, 0)

    assert mask.width_px == 16
    assert mask.height_px == 16
    assert mask.opening_pixel_count == 49
    assert center.material_id == "Si"
    assert center.is_opening is True
    assert corner.material_id == "PR"
    assert corner.is_opening is False


def test_inspect_geometry_cli_reports_click_material() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "inspect_geometry.py"),
            "--scene",
            str(FIXTURE_ROOT / "scenes" / "pr_hole_scene.json"),
            "--click-nm",
            "0,0,1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "geometry_ok=true" in result.stdout
    assert "feature_type=hole" in result.stdout
    assert "click_material=Si" in result.stdout
    assert "click_region=opening" in result.stdout


def test_inspect_geometry_cli_reports_png_mask_summary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "inspect_geometry.py"),
            "--mask",
            str(FIXTURE_ROOT / "geometry" / "pr_trench.png"),
            "--pixel",
            "7,4",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mask_ok=true" in result.stdout
    assert "opening_pixels=48" in result.stdout
    assert "pixel_material=Si" in result.stdout
