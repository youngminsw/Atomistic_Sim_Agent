from __future__ import annotations

from collections.abc import Mapping, Sequence

from .errors import SchemaValidationError


JsonMap = Mapping[str, object]


def as_mapping(value: object, field: str) -> JsonMap:
    if isinstance(value, Mapping):
        return value
    raise SchemaValidationError(f"{field} must be an object")


def as_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, list | tuple):
        return value
    raise SchemaValidationError(f"{field} must be a list")


def require(mapping: JsonMap, field: str, message: str | None = None) -> object:
    if field not in mapping or mapping[field] is None:
        raise SchemaValidationError(message or f"{field} required")
    return mapping[field]


def as_str(value: object, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise SchemaValidationError(f"{field} must be a non-empty string")


def as_float(value: object, field: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise SchemaValidationError(f"{field} must be a number")


def as_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be a boolean")


def str_field(mapping: JsonMap, field: str) -> str:
    return as_str(require(mapping, field), field)


def float_field(mapping: JsonMap, field: str) -> float:
    return as_float(require(mapping, field), field)


def optional_str(mapping: JsonMap, field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    return as_str(value, field)


def float_map(value: object, field: str) -> dict[str, float]:
    mapping = as_mapping(value, field)
    return {as_str(key, field): as_float(val, f"{field}.{key}") for key, val in mapping.items()}
