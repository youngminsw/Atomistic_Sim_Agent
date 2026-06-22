from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import assess_md_production_acceptance
from sim_agent.schemas._parse import JsonMap, as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-incidents", type=int, default=500)
    args = parser.parse_args()

    try:
        payload = _read_mapping(Path(args.postprocess_report), "md_postprocess_report")
        report = assess_md_production_acceptance(payload, args.minimum_incidents)
        _write_json(Path(args.out), report.payload)
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print(f"md_production_acceptance={str(report.accepted).lower()}")
    print(f"minimum_incidents={report.payload['minimum_incidents']}")
    print(f"event_count={report.payload['event_count']}")
    if report.payload["blockers"]:
        print(f"blockers={','.join(report.payload['blockers'])}")
    if report.payload["evidence"]:
        print(f"evidence={','.join(report.payload['evidence'])}")
    return 0 if report.accepted else 1


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
