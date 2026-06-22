from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import postprocess_lammps_execution_result, write_lammps_execution_ledger
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-result", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--events-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--worker-capability", default="")
    parser.add_argument("--ledger-dir", default="")
    parser.add_argument("--required-ion")
    args = parser.parse_args()

    try:
        execution_result = _read_mapping(
            Path(args.execution_result),
            "lammps_execution_result",
        )
        report = postprocess_lammps_execution_result(
            execution_result,
            material_id=args.material,
            descriptor_root=Path(args.descriptor_root),
            events_out=Path(args.events_out),
            required_ion=args.required_ion,
        )
        report_path = Path(args.report_out)
        events_path = Path(args.events_out)
        _write_json(report_path, report.payload)
        ledger_path = _write_ledger(
            ledger_dir=args.ledger_dir,
            worker_capability_path=args.worker_capability,
            execution_result=execution_result,
            postprocess_report=report.payload,
            events_path=events_path,
        )
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    status = as_str(require(report.payload, "postprocess_status"), "postprocess_status")
    print(f"md_postprocess_ok={str(report.ok).lower()}")
    print(f"postprocess_status={status}")
    print(f"event_count={report.payload['event_count']}")
    print(f"verification_status={report.payload['verification_status']}")
    if report.payload["evidence"]:
        print(f"evidence={','.join(report.payload['evidence'])}")
    if report.payload["errors"]:
        print(f"errors={','.join(report.payload['errors'])}")
    if ledger_path is not None:
        print(f"lammps_execution_ledger_path={ledger_path}")
    return 0 if report.ok else 1


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ledger(
    ledger_dir: str,
    worker_capability_path: str,
    execution_result: JsonMap,
    postprocess_report: JsonMap,
    events_path: Path,
) -> Path | None:
    if not ledger_dir:
        return None
    if not worker_capability_path:
        raise SchemaValidationError("worker_capability_required_for_ledger")
    worker_capability = _read_mapping(Path(worker_capability_path), "worker_capability")
    bundle = write_lammps_execution_ledger(
        output_dir=Path(ledger_dir),
        run_id=as_str(require(postprocess_report, "run_id"), "run_id"),
        worker_capability_payload=worker_capability,
        execution_result_payload=execution_result,
        postprocess_report_payload=postprocess_report,
        events_path=events_path,
    )
    return bundle.ledger_path


if __name__ == "__main__":
    raise SystemExit(main())
