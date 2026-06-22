from __future__ import annotations

from pathlib import Path

from .types import Bounds3D, GeometryError


def load_ascii_stl_bounds(path: Path) -> Bounds3D:
    vertices = tuple(_iter_vertices(path))
    if not vertices:
        raise GeometryError("stl_contains_no_vertices")
    xs = tuple(point[0] for point in vertices)
    ys = tuple(point[1] for point in vertices)
    zs = tuple(point[2] for point in vertices)
    return Bounds3D(
        x_min_nm=min(xs),
        x_max_nm=max(xs),
        y_min_nm=min(ys),
        y_max_nm=max(ys),
        z_min_nm=min(zs),
        z_max_nm=max(zs),
    )


def _iter_vertices(path: Path) -> tuple[tuple[float, float, float], ...]:
    points: list[tuple[float, float, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("vertex "):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise GeometryError("invalid_stl_vertex")
        points.append((_float(parts[1]), _float(parts[2]), _float(parts[3])))
    return tuple(points)


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise GeometryError(f"invalid_float={raw}") from exc
