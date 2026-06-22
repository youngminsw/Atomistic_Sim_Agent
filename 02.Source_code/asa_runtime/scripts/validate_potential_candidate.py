from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.materials import validate_potential_candidate
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--ion", required=True)
    parser.add_argument("--required-elements", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        candidate = _read_mapping(Path(args.candidate), "potential_candidate")
        report = validate_potential_candidate(
            candidate,
            material_id=args.material,
            ion_species=args.ion,
            required_elements=_csv_tuple(args.required_elements),
        )
        _write_json(Path(args.out), report.payload)
        gate_status = as_str(require(report.payload, "gate_status"), "gate_status")
        potential_id = as_str(require(report.payload, "potential_id"), "potential_id")
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print(f"potential_gate_ok={str(report.ok).lower()}")
    print(f"gate_status={gate_status}")
    print(f"potential_id={potential_id}")
    return 0 if report.ok else 1


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
