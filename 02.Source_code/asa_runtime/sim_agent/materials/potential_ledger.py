from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require

from .potential_acquisition import PotentialAcquisitionReport


ARTIFACT_COUNT: Final = 5


@dataclass(frozen=True, slots=True)
class PotentialAcquisitionLedgerBundle:
    output_dir: Path
    ledger_path: Path
    report_path: Path
    candidate_path: Path
    validation_path: Path
    sandbox_smoke_path: Path

    @property
    def artifact_count(self) -> int:
        return ARTIFACT_COUNT


def write_potential_acquisition_ledger(
    output_dir: Path,
    run_id: str,
    report: PotentialAcquisitionReport,
) -> PotentialAcquisitionLedgerBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    sandbox = as_mapping(require(report.payload, "sandbox_smoke"), "sandbox_smoke")
    validation = as_mapping(require(report.payload, "validation"), "validation")

    report_path = output_dir / "potential_acquisition_report.json"
    candidate_path = output_dir / "potential_candidate.json"
    validation_path = output_dir / "potential_validation_report.json"
    sandbox_smoke_path = output_dir / "sandbox_smoke_report.json"
    ledger_path = output_dir / "ledger.json"

    sandbox_input = _artifact_path(output_dir, sandbox, "input_path")
    sandbox_potential = _artifact_path(output_dir, sandbox, "potential_path")
    if report.ok:
        _require_existing_artifact(output_dir, sandbox_input)
        _require_existing_artifact(output_dir, sandbox_potential)

    _write_json(report_path, report.payload)
    _write_json(candidate_path, report.candidate_payload)
    _write_json(validation_path, validation)
    _write_json(sandbox_smoke_path, sandbox)
    _write_json(
        ledger_path,
        _ledger_payload(
            run_id=run_id,
            report=report,
            sandbox=sandbox,
            sandbox_input=sandbox_input,
            sandbox_potential=sandbox_potential,
        ),
    )

    return PotentialAcquisitionLedgerBundle(
        output_dir=output_dir,
        ledger_path=ledger_path,
        report_path=report_path,
        candidate_path=candidate_path,
        validation_path=validation_path,
        sandbox_smoke_path=sandbox_smoke_path,
    )


def _ledger_payload(
    run_id: str,
    report: PotentialAcquisitionReport,
    sandbox: JsonMap,
    sandbox_input: str,
    sandbox_potential: str,
) -> JsonMap:
    gate_status = as_str(require(report.payload, "gate_status"), "gate_status")
    evidence = require(report.payload, "acquisition_evidence")
    return {
        "run_id": run_id,
        "run_status": "complete" if report.ok else "failed",
        "gate_status": gate_status,
        "artifact_count": ARTIFACT_COUNT,
        "artifact_types": [
            "potential_acquisition_report",
            "potential_candidate",
            "potential_validation_report",
            "potential_sandbox_smoke_report",
            "potential_sandbox_smoke_run",
        ],
        "artifacts": {
            "ledger": "ledger.json",
            "acquisition_report": "potential_acquisition_report.json",
            "candidate": "potential_candidate.json",
            "validation_report": "potential_validation_report.json",
            "sandbox_smoke_report": "sandbox_smoke_report.json",
            "sandbox_input": sandbox_input,
            "sandbox_potential": sandbox_potential,
        },
        "sandbox_smoke": sandbox,
        "verification_evidence": evidence,
    }


def _artifact_path(output_dir: Path, payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _require_existing_artifact(output_dir: Path, artifact_path: str) -> None:
    if not artifact_path:
        raise FileNotFoundError("sandbox_smoke_artifact_missing")
    path = Path(artifact_path)
    candidate = path if path.is_absolute() else output_dir / path
    if not candidate.exists():
        raise FileNotFoundError("sandbox_smoke_artifact_missing")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
