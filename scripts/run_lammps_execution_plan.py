from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import LAMMPSExecutionRunError, run_lammps_execution_plan
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker-capability", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    try:
        plan = _read_mapping(Path(args.plan), "lammps_execution_plan")
        result = run_lammps_execution_plan(
            plan,
            execute_now=args.execute,
            worker_capability_path=_optional_path(args.worker_capability),
        )
        _write_json(Path(args.out), result.manifest_payload)
    except (
        OSError,
        json.JSONDecodeError,
        SchemaValidationError,
        LAMMPSExecutionRunError,
    ) as exc:
        print(str(exc))
        return 1

    status = as_str(require(result.manifest_payload, "execution_status"), "execution_status")
    print("lammps_execution_runner_ok=true")
    print(f"execution_status={status}")
    print(
        "worker_capability_gate_status="
        f"{result.manifest_payload['worker_capability_gate_status']}"
    )
    print(f"working_directory={result.working_directory}")
    print(f"command_line={result.manifest_payload['command_line']}")
    if args.execute and status != "lammps_completed":
        return 1
    return 0


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_path(value: str) -> Path | None:
    if not value:
        return None
    return Path(value)


if __name__ == "__main__":
    raise SystemExit(main())
