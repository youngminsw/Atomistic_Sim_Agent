from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.materials import (
    PotentialAcquisitionError,
    PotentialAcquisitionReport,
    PotentialAcquisitionRequest,
    acquire_potential_candidate,
    write_potential_acquisition_ledger,
)
from sim_agent.schemas._parse import JsonMap, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--metadata-url", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--ion", required=True)
    parser.add_argument("--required-elements", required=True)
    parser.add_argument("--lammps-command", default="")
    parser.add_argument("--sandbox-work-dir", default="")
    parser.add_argument("--ledger-dir", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default=str(SOURCE_ROOT))
    args = parser.parse_args()

    try:
        report = acquire_potential_candidate(
            PotentialAcquisitionRequest(
                source_url=args.source_url,
                metadata_url=args.metadata_url,
                material_id=args.material,
                ion_species=args.ion,
                required_elements=_csv_tuple(args.required_elements),
                sandbox_command=_command_tuple(args.lammps_command),
                sandbox_work_dir=_sandbox_work_dir(args.sandbox_work_dir, args.ledger_dir),
            ),
            repo_root=Path(args.repo_root),
        )
        _write_json(Path(args.out), report.payload)
        ledger_path = _write_ledger(args.ledger_dir, report)
        gate_status = as_str(require(report.payload, "gate_status"), "gate_status")
    except (OSError, SchemaValidationError, PotentialAcquisitionError) as exc:
        print(str(exc))
        return 1

    print(f"potential_acquisition_ok={str(report.ok).lower()}")
    print(f"gate_status={gate_status}")
    if ledger_path is not None:
        print(f"ledger_path={ledger_path}")
    return 0 if report.ok else 1


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _command_tuple(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(shlex.split(value))


def _optional_path(value: str) -> Path | None:
    if not value:
        return None
    return Path(value)


def _sandbox_work_dir(sandbox_work_dir: str, ledger_dir: str) -> Path | None:
    explicit = _optional_path(sandbox_work_dir)
    if explicit is not None:
        return explicit
    ledger = _optional_path(ledger_dir)
    if ledger is None:
        return None
    return ledger / "sandbox_smoke"


def _write_ledger(ledger_dir: str, report: PotentialAcquisitionReport) -> Path | None:
    path = _optional_path(ledger_dir)
    if path is None:
        return None
    ledger = write_potential_acquisition_ledger(
        output_dir=path,
        run_id="potential-acquisition",
        report=report,
    )
    return ledger.ledger_path


if __name__ == "__main__":
    raise SystemExit(main())
