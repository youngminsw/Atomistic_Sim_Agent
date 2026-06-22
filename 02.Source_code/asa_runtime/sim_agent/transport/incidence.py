from __future__ import annotations

import math

from .types import SurfaceNormal3D, TransportError


def local_incidence_angle_deg(polar_deg: float, azimuth_deg: float, normal: SurfaceNormal3D) -> float:
    normal_length = math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
    if normal_length <= 1.0e-12:
        raise TransportError("surface_normal_must_be_nonzero")
    polar_rad = math.radians(polar_deg)
    azimuth_rad = math.radians(azimuth_deg)
    direction_x = math.sin(polar_rad) * math.cos(azimuth_rad)
    direction_y = math.sin(polar_rad) * math.sin(azimuth_rad)
    direction_z = -math.cos(polar_rad)
    dot = abs(
        direction_x * (normal.x / normal_length)
        + direction_y * (normal.y / normal_length)
        + direction_z * (normal.z / normal_length)
    )
    return math.degrees(math.acos(min(max(dot, 0.0), 1.0)))
