from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .parser import parse_lammps_output_run
from .verification import verify_md_run


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionPostprocessReport:
    ok: bool
    payload: JsonMap


@dataclass(frozen=True, slots=True)
class _ExecutionResult:
    execution_result_id: str
    run_id: str
    execution_status: str
    working_directory: Path
    missing_expected_outputs: tuple[str, ...]
    return_code: int | None
    expected_incident_count: int | None


@dataclass(frozen=True, slots=True)
class _GateFailure:
    postprocess_status: str
    errors: tuple[str, ...]


def postprocess_lammps_execution_result(
    execution_result_payload: JsonMap,
    material_id: str,
    descriptor_root: Path,
    events_out: Path,
    required_ion: str | None = None,
) -> LAMMPSExecutionPostprocessReport:
    result = _execution_result(execution_result_payload)
    gate_failure = _gate_failure(result)
    if gate_failure is not None:
        return LAMMPSExecutionPostprocessReport(
            ok=False,
            payload=_payload(
                result,
                events_out,
                gate_failure.postprocess_status,
                False,
                0,
                0,
                0.0,
                "not_run",
                (),
                gate_failure.errors,
            ),
        )

    parse_report = parse_lammps_output_run(
        run_dir=result.working_directory,
        material_id=material_id,
        descriptor_root=descriptor_root,
        out_path=events_out,
    )
    if not parse_report.ok:
        return LAMMPSExecutionPostprocessReport(
            ok=False,
            payload=_payload(
                result,
                events_out,
                "md_parse_failed",
                False,
                0,
                0,
                0.0,
                "not_run",
                parse_report.evidence,
                parse_report.errors,
            ),
        )

    verification = verify_md_run(
        log_path=result.working_directory / "log.lammps",
        events_path=events_out,
        expected_events=result.expected_incident_count or parse_report.event_count,
        required_ion=required_ion,
        required_material=material_id,
    )
    if not verification.ok:
        return LAMMPSExecutionPostprocessReport(
            ok=False,
            payload=_payload(
                result,
                events_out,
                "md_verification_failed",
                False,
                parse_report.event_count,
                parse_report.layer_removed_count,
                parse_report.total_deposited_energy_eV,
                verification.status.value,
                parse_report.evidence + verification.evidence,
                verification.errors,
            ),
        )

    return LAMMPSExecutionPostprocessReport(
        ok=True,
        payload=_payload(
            result,
            events_out,
            "md_postprocess_complete",
            True,
            parse_report.event_count,
            parse_report.layer_removed_count,
            parse_report.total_deposited_energy_eV,
            verification.status.value,
            parse_report.evidence + verification.evidence,
            (),
        ),
    )


def _execution_result(payload: JsonMap) -> _ExecutionResult:
    return _ExecutionResult(
        execution_result_id=as_str(
            require(payload, "execution_result_id"),
            "execution_result_id",
        ),
        run_id=as_str(require(payload, "run_id"), "run_id"),
        execution_status=as_str(require(payload, "execution_status"), "execution_status"),
        working_directory=Path(as_str(require(payload, "working_directory"), "working_directory")),
        missing_expected_outputs=_string_tuple(payload, "missing_expected_outputs"),
        return_code=_optional_int(payload, "return_code"),
        expected_incident_count=_optional_positive_int(payload, "expected_incident_count"),
    )


def _gate_failure(result: _ExecutionResult) -> _GateFailure | None:
    match result.execution_status:
        case "lammps_completed":
            pass
        case status:
            return _GateFailure(
                postprocess_status="lammps_execution_not_complete",
                errors=(f"lammps_execution_status={status}",),
            )
    if result.return_code is None:
        return _GateFailure(
            postprocess_status="lammps_execution_not_complete",
            errors=("lammps_return_code_missing",),
        )
    if result.return_code != 0:
        return _GateFailure(
            postprocess_status="lammps_execution_not_complete",
            errors=(f"lammps_return_code={result.return_code}",),
        )
    if result.missing_expected_outputs:
        return _GateFailure(
            postprocess_status="lammps_outputs_missing",
            errors=tuple(
                f"missing_expected_output={path}"
                for path in result.missing_expected_outputs
            ),
        )
    return None


def _payload(
    result: _ExecutionResult,
    events_out: Path,
    postprocess_status: str,
    ok: bool,
    event_count: int,
    layer_removed_count: int,
    total_deposited_energy_eV: float,
    verification_status: str,
    evidence: tuple[str, ...],
    errors: tuple[str, ...],
) -> JsonMap:
    return {
        "postprocess_result_id": f"{result.run_id}-md-postprocess",
        "execution_result_id": result.execution_result_id,
        "run_id": result.run_id,
        "postprocess_status": postprocess_status,
        "ok": ok,
        "working_directory": str(result.working_directory),
        "events_path": str(events_out),
        "event_count": event_count,
        "expected_incident_count": result.expected_incident_count,
        "layer_removed_count": layer_removed_count,
        "total_deposited_energy_eV": total_deposited_energy_eV,
        "verification_status": verification_status,
        "evidence": list(evidence),
        "errors": list(errors),
    }


def _string_tuple(payload: JsonMap, field: str) -> tuple[str, ...]:
    return tuple(as_str(item, field) for item in as_sequence(require(payload, field), field))


def _optional_int(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer or null")


def _optional_positive_int(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise SchemaValidationError(f"{field} must be a positive integer or null")
