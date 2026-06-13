from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_bool, as_str, require


ARTIFACT_COUNT: Final = 5


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionLedgerBundle:
    output_dir: Path
    ledger_path: Path
    worker_capability_path: Path
    execution_result_path: Path
    postprocess_report_path: Path
    events_path: Path

    @property
    def artifact_count(self) -> int:
        return ARTIFACT_COUNT


def write_lammps_execution_ledger(
    output_dir: Path,
    run_id: str,
    worker_capability_payload: JsonMap,
    execution_result_payload: JsonMap,
    postprocess_report_payload: JsonMap,
    events_path: Path,
) -> LAMMPSExecutionLedgerBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_events_path = output_dir / "md_events.jsonl"
    if as_bool(require(postprocess_report_payload, "ok"), "ok"):
        _require_existing_file(events_path, "md_events_missing")
    _copy_file(events_path, target_events_path)

    worker_path = output_dir / "worker_capability.json"
    execution_path = output_dir / "lammps_execution_result.json"
    postprocess_path = output_dir / "md_postprocess_report.json"
    ledger_path = output_dir / "ledger.json"

    _write_json(worker_path, worker_capability_payload)
    _write_json(execution_path, execution_result_payload)
    _write_json(postprocess_path, postprocess_report_payload)
    _write_json(
        ledger_path,
        _ledger_payload(
            run_id,
            worker_capability_payload,
            execution_result_payload,
            postprocess_report_payload,
        ),
    )
    return LAMMPSExecutionLedgerBundle(
        output_dir=output_dir,
        ledger_path=ledger_path,
        worker_capability_path=worker_path,
        execution_result_path=execution_path,
        postprocess_report_path=postprocess_path,
        events_path=target_events_path,
    )


def _ledger_payload(
    run_id: str,
    worker_capability: JsonMap,
    execution_result: JsonMap,
    postprocess_report: JsonMap,
) -> JsonMap:
    postprocess_ok = as_bool(require(postprocess_report, "ok"), "ok")
    return {
        "run_id": run_id,
        "run_status": "complete" if postprocess_ok else "failed",
        "artifact_count": ARTIFACT_COUNT,
        "artifact_types": [
            "lammps_execution_ledger",
            "worker_capability",
            "lammps_execution_result",
            "md_postprocess_report",
            "md_events",
        ],
        "artifacts": {
            "ledger": "ledger.json",
            "worker_capability": "worker_capability.json",
            "lammps_execution_result": "lammps_execution_result.json",
            "md_postprocess_report": "md_postprocess_report.json",
            "md_events": "md_events.jsonl",
        },
        "worker_capability_gate_status": _text(worker_capability, "gate_status"),
        "execution_status": _text(execution_result, "execution_status"),
        "postprocess_status": _text(postprocess_report, "postprocess_status"),
        "verification_status": _text(postprocess_report, "verification_status"),
        "expected_incident_count": _int_value(postprocess_report, "expected_incident_count"),
        "event_count": _int_value(postprocess_report, "event_count"),
        "layer_removed_count": _int_value(postprocess_report, "layer_removed_count"),
        "total_deposited_energy_eV": _float_value(
            postprocess_report,
            "total_deposited_energy_eV",
        ),
        "verification_evidence": _sequence_strings(postprocess_report, "evidence"),
        "preflight_evidence": _sequence_strings(execution_result, "preflight_evidence"),
        "worker_evidence": _sequence_strings(worker_capability, "evidence"),
        "errors": _sequence_strings(postprocess_report, "errors"),
    }


def _copy_file(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    shutil.copyfile(source, target)


def _require_existing_file(path: Path, code: str) -> None:
    if not path.exists():
        raise FileNotFoundError(code)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def _int_value(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _float_value(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _sequence_strings(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]
