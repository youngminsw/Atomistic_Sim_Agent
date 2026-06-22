from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from .types import GeometryError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class PixelIndex:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class MaskPixel:
    x: int
    y: int
    value: int
    material_id: str
    is_opening: bool


@dataclass(frozen=True, slots=True)
class BinaryMask2D:
    path: str
    width_px: int
    height_px: int
    values: tuple[int, ...]
    opening_pixels: frozenset[PixelIndex]
    target_material_id: str
    mask_material_id: str

    @property
    def opening_pixel_count(self) -> int:
        return len(self.opening_pixels)

    def pixel_at(self, x: int, y: int) -> MaskPixel:
        if x < 0 or x >= self.width_px or y < 0 or y >= self.height_px:
            raise GeometryError("pixel_outside_mask_bounds")
        value = self.values[y * self.width_px + x]
        is_opening = PixelIndex(x, y) in self.opening_pixels
        material_id = self.target_material_id if is_opening else self.mask_material_id
        return MaskPixel(x=x, y=y, value=value, material_id=material_id, is_opening=is_opening)


def load_png_mask(
    path: Path,
    target_material_id: str,
    mask_material_id: str,
    threshold: int = 127,
) -> BinaryMask2D:
    data = path.read_bytes()
    width_px, height_px, compressed = _read_png_chunks(data)
    values = _decompress_grayscale_rows(compressed, width_px, height_px)
    opening_pixels = frozenset(
        PixelIndex(index % width_px, index // width_px) for index, value in enumerate(values) if value > threshold
    )
    return BinaryMask2D(
        path=str(path),
        width_px=width_px,
        height_px=height_px,
        values=values,
        opening_pixels=opening_pixels,
        target_material_id=target_material_id,
        mask_material_id=mask_material_id,
    )


def _read_png_chunks(data: bytes) -> tuple[int, int, bytes]:
    if data[:8] != PNG_SIGNATURE:
        raise GeometryError("expected_png_mask")
    position = 8
    width_px = 0
    height_px = 0
    compressed = b""
    while position < len(data):
        chunk_length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + chunk_length]
        position += chunk_length + 12
        if chunk_type == b"IHDR":
            width_px, height_px, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", payload)
            if bit_depth != 8 or color_type != 0:
                raise GeometryError("only_8bit_grayscale_png_masks_supported")
        if chunk_type == b"IDAT":
            compressed += payload
    if width_px <= 0 or height_px <= 0 or not compressed:
        raise GeometryError("invalid_png_mask_chunks")
    return width_px, height_px, compressed


def _decompress_grayscale_rows(compressed: bytes, width_px: int, height_px: int) -> tuple[int, ...]:
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise GeometryError("invalid_png_idat") from exc
    stride = width_px + 1
    expected = stride * height_px
    if len(raw) != expected:
        raise GeometryError("unexpected_png_row_length")
    values: list[int] = []
    for row_index in range(height_px):
        row_start = row_index * stride
        filter_type = raw[row_start]
        if filter_type != 0:
            raise GeometryError("png_filter_not_supported")
        values.extend(raw[row_start + 1 : row_start + stride])
    return tuple(values)
