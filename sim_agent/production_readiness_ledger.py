from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping


SCRIPT_ROOT = "02.Source_code/asa_runtime/scripts"


def read_json(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def read_optional_json(path: Path | None) -> JsonMap:
    if path is None:
        return {}
    try:
        return read_json(path, path.name)
    except (OSError, json.JSONDecodeError):
        return {}


def mapping(payload: JsonMap, field: str) -> JsonMap:
    value = payload.get(field)
    if isinstance(value, dict):
        return value
    return {}


def text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def string_list(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def positive_int(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def int_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def artifact_path(ledger: JsonMap, field: str) -> str:
    artifact_paths = mapping(ledger, "artifact_paths")
    return text(artifact_paths, field)


def artifact_output(ledger: JsonMap, filename: str) -> str:
    return str(Path(artifact_dir(ledger)) / filename)


def artifact_dir(ledger: JsonMap) -> str:
    return text(ledger, "artifact_dir") or "."


def missing_fields(fields: tuple[tuple[str, str], ...]) -> list[str]:
    return [name for name, value in fields if not value]


def amorphous_prep_worker_present(ledger: JsonMap) -> bool:
    artifact_paths = mapping(ledger, "artifact_paths")
    return bool(text(artifact_paths, "amorphous_structure_prep_worker_path"))


def amorphous_prep_status(ledger: JsonMap) -> str:
    remote = mapping(ledger, "remote")
    return text(remote, "amorphous_prep_status")


def amorphous_prep_blockers(ledger: JsonMap) -> list[str]:
    remote = mapping(ledger, "remote")
    return string_list(remote, "amorphous_prep_blockers")


def metric_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value:
        return value
    return ""


def range_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        minimum = metric_text(value, "minimum")
        maximum = metric_text(value, "maximum")
        if minimum and maximum:
            return f"{minimum}:{maximum}"
    return ""
