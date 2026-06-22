from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import assess_md_physics_readiness
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--incident-schedule", required=True)
    parser.add_argument("--surface-state", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        report = assess_md_physics_readiness(
            _read_mapping(Path(args.manifest), "manifest"),
            _read_mapping(Path(args.contract), "contract"),
            _read_mapping(Path(args.incident_schedule), "incident_schedule"),
            _read_mapping(Path(args.surface_state), "surface_state"),
        )
        _write_json(Path(args.out), report.payload)
        gate_status = as_str(require(report.payload, "gate_status"), "gate_status")
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print(f"md_physics_gate_ok={str(report.ok).lower()}")
    print(f"gate_status={gate_status}")
    print(f"production_ready={str(report.production_ready).lower()}")
    return 0 if report.ok else 1


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
