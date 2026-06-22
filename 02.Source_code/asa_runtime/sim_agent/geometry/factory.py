from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, as_sequence, str_field

from .stl import load_ascii_stl_bounds
from .types import GeometryError, GridShape, PatternGeometry3D, parse_feature_type


def load_pattern_geometry_from_scene(
    scene: JsonMap,
    source_root: Path,
    grid_shape: GridShape,
    target_depth_nm: float,
) -> PatternGeometry3D:
    geometry_source = as_mapping(scene.get("geometry_source"), "geometry_source")
    source_path = _resolve_source_path(source_root, str_field(geometry_source, "path"))
    bounds = load_ascii_stl_bounds(source_path).with_min_depth(target_depth_nm)
    return PatternGeometry3D(
        feature_type=parse_feature_type(str_field(scene, "feature_type")),
        bounds=bounds,
        grid_shape=grid_shape,
        source_path=str(source_path),
        target_material_id=_target_material(scene),
        mask_material_id=_mask_material(scene),
        pr_selectivity=_pr_selectivity(scene),
    )


def _resolve_source_path(source_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    rooted = source_root / candidate
    if rooted.exists():
        return rooted
    test_rooted = source_root / "tests" / candidate
    if test_rooted.exists():
        return test_rooted
    raise GeometryError(f"geometry_source_not_found={raw_path}")


def _target_material(scene: JsonMap) -> str:
    value = scene.get("target_material")
    if isinstance(value, str) and value:
        return value
    surface_state = scene.get("surface_state")
    if isinstance(surface_state, dict):
        return str_field(surface_state, "material_id")
    return _material_by_role(scene, "target", "target")


def _mask_material(scene: JsonMap) -> str:
    value = scene.get("mask_material")
    if isinstance(value, str) and value:
        return value
    return _material_by_role(scene, "mask", "PR")


def _pr_selectivity(scene: JsonMap) -> float:
    value = scene.get("pr_selectivity")
    if value is not None:
        return as_float(value, "pr_selectivity")
    material = _material_mapping_by_role(scene, "mask")
    selectivity = material.get("pr_selectivity")
    if selectivity is None:
        return 1.0
    return as_float(selectivity, "pr_selectivity")


def _material_by_role(scene: JsonMap, role: str, fallback: str) -> str:
    material = _material_mapping_by_role(scene, role)
    material_id = material.get("material_id")
    if isinstance(material_id, str) and material_id:
        return material_id
    return fallback


def _material_mapping_by_role(scene: JsonMap, role: str) -> JsonMap:
    material_stack = scene.get("material_stack")
    if not isinstance(material_stack, dict):
        return {}
    materials = material_stack.get("materials", ())
    if not isinstance(materials, Sequence):
        return {}
    for item in as_sequence(materials, "materials"):
        material = as_mapping(item, "materials[]")
        if material.get("role") == role:
            return material
    return {}
