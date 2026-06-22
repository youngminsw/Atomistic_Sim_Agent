from __future__ import annotations

import json
import time
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


def read_jsonl(path: Path) -> list[JsonMap] | None:
    if not path.is_file():
        return []
    records: list[JsonMap] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                return None
            records.append(value)
    except json.JSONDecodeError:
        return None
    return records


def read_json(path: Path) -> JsonMap | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: JsonMap) -> None:
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
