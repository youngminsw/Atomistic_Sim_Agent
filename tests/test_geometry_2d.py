from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "geometry"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_2d_pattern_image_converts_to_material_grid_sdf_and_normals(tmp_path: Path) -> None:
    from sim_agent.geometry import load_pattern_geometry_2d

    geometry = load_pattern_geometry_2d(
        image_path=FIXTURE_ROOT / "pr_trench.png",
        pixel_size_nm=2.0,
        target_material_id="Si",
        mask_material_id="PR",
        structure_description="2D PR trench mask",
    )
    preview = geometry.write_preview_ppm(tmp_path / "preview.ppm")
    center = geometry.cell_at_nm(14.0, 8.0)
    side = geometry.cell_at_nm(2.0, 8.0)
    boundary = geometry.cell_at_nm(8.0, 8.0)

    assert geometry.width_px == 16
    assert geometry.height_px == 16
    assert geometry.width_nm == 32.0
    assert geometry.height_nm == 32.0
    assert geometry.structure_description == "2D PR trench mask"
    assert center.material_id == "Si"
    assert center.region == "opening"
    assert center.signed_distance_nm > 0.0
    assert side.material_id == "PR"
    assert side.region == "mask"
    assert side.signed_distance_nm < 0.0
    assert abs(boundary.normal.x * boundary.normal.x + boundary.normal.y * boundary.normal.y - 1.0) < 1.0e-6
    assert preview.exists()
    assert preview.read_text(encoding="ascii").startswith("P3\n16 16\n255\n")


def test_2d_pattern_image_requires_pixel_size() -> None:
    from sim_agent.geometry import GeometryError, load_pattern_geometry_2d

    try:
        load_pattern_geometry_2d(
            image_path=FIXTURE_ROOT / "pr_trench.png",
            pixel_size_nm=0.0,
            target_material_id="Si",
            mask_material_id="PR",
            structure_description="2D PR trench mask",
        )
    except GeometryError as exc:
        assert str(exc) == "pixel_size_required"
    else:
        raise AssertionError("expected GeometryError")


def test_smoke_geometry_cli_reports_2d_grid_and_preview(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_geometry.py"),
            "--mode",
            "2d",
            "--fixture",
            str(FIXTURE_ROOT / "pr_trench.png"),
            "--pixel-size-nm",
            "2.0",
            "--preview",
            str(tmp_path / "preview.ppm"),
            "--click-nm",
            "14,8",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "geometry_valid=true" in result.stdout
    assert "mode=2d" in result.stdout
    assert "materials=PR,Si" in result.stdout
    assert "sdf_written=true" in result.stdout
    assert "normals_valid=true" in result.stdout
    assert "click_material=Si" in result.stdout


def test_smoke_geometry_cli_rejects_2d_image_without_scale() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_geometry.py"),
            "--mode",
            "2d",
            "--fixture",
            str(FIXTURE_ROOT / "pr_trench.png"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pixel_size_required" in result.stdout
