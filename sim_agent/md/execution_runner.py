from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import (
    JsonMap,
    as_bool,
    as_sequence,
    as_str,
    require,
)
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionRunError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionResult:
    working_directory: Path
    command: tuple[str, ...]
    manifest_payload: JsonMap


@dataclass(frozen=True, slots=True)
class _RunnablePlan:
    execution_plan_id: str
    run_id: str
    execution_status: str
    preflight_ok: bool
    working_directory: Path
    lammps_binary: str
    input_deck: str
    expected_incident_count: int | None
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class _WorkerCapabilityEvidence:
    path: Path | None
    gate_status: str
    evidence: tuple[str, ...]


def run_lammps_execution_plan(
    execution_plan_payload: JsonMap,
    execute_now: bool,
    worker_capability_path: Path | None = None,
) -> LAMMPSExecutionResult:
    try:
        plan = _runnable_plan(execution_plan_payload)
    except SchemaValidationError as exc:
        raise LAMMPSExecutionRunError(str(exc)) from exc
    capability = _worker_capability_evidence(worker_capability_path)
    _ensure_ready(plan)
    command = (plan.lammps_binary, "-in", plan.input_deck)
    if not execute_now:
        return LAMMPSExecutionResult(
            working_directory=plan.working_directory,
            command=command,
            manifest_payload=_manifest_payload(plan, command, execute_now, capability, None, ()),
        )
    outcome = _run_process(plan.working_directory, command)
    missing_outputs = _missing_files(plan.working_directory, plan.expected_outputs)
    return LAMMPSExecutionResult(
        working_directory=plan.working_directory,
        command=command,
        manifest_payload=_manifest_payload(
            plan,
            command,
            execute_now,
            capability,
            outcome,
            missing_outputs,
        ),
    )


def _runnable_plan(payload: JsonMap) -> _RunnablePlan:
    return _RunnablePlan(
        execution_plan_id=as_str(require(payload, "execution_plan_id"), "execution_plan_id"),
        run_id=as_str(require(payload, "run_id"), "run_id"),
        execution_status=as_str(require(payload, "execution_status"), "execution_status"),
        preflight_ok=as_bool(require(payload, "preflight_ok"), "preflight_ok"),
        working_directory=Path(as_str(require(payload, "working_directory"), "working_directory")),
        lammps_binary=as_str(require(payload, "lammps_binary"), "lammps_binary"),
        input_deck=_relative_file(
            as_str(require(payload, "input_deck"), "input_deck"),
            "input_deck",
        ),
        expected_incident_count=_optional_positive_int(payload, "expected_incident_count"),
        required_inputs=_relative_file_tuple(payload, "required_inputs"),
        expected_outputs=_relative_file_tuple(payload, "expected_outputs"),
    )


def _relative_file_tuple(payload: JsonMap, field: str) -> tuple[str, ...]:
    return tuple(
        _relative_file(as_str(item, field), field)
        for item in as_sequence(require(payload, field), field)
    )


def _relative_file(raw: str, field: str) -> str:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaValidationError(f"{field} must be a relative run file")
    return raw


def _ensure_ready(plan: _RunnablePlan) -> None:
    if plan.execution_status != "ready_for_lammps":
        raise LAMMPSExecutionRunError(f"lammps_plan_not_ready={plan.execution_status}")
    if not plan.preflight_ok:
        raise LAMMPSExecutionRunError("lammps_plan_preflight_not_ok")
    missing_inputs = _missing_files(plan.working_directory, plan.required_inputs)
    if missing_inputs:
        raise LAMMPSExecutionRunError(f"lammps_required_input_missing={missing_inputs[0]}")


def _run_process(working_directory: Path, command: tuple[str, ...]) -> _ProcessOutcome:
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise LAMMPSExecutionRunError(f"lammps_launch_failed={command[0]}") from exc
    return _ProcessOutcome(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _missing_files(run_dir: Path, filenames: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(filename for filename in filenames if not (run_dir / filename).exists())


def _worker_capability_evidence(path: Path | None) -> _WorkerCapabilityEvidence:
    if path is None:
        return _WorkerCapabilityEvidence(
            path=None,
            gate_status="worker_capability_not_provided",
            evidence=(),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LAMMPSExecutionRunError("worker_capability_report_missing") from exc
    except json.JSONDecodeError as exc:
        raise LAMMPSExecutionRunError("worker_capability_report_invalid_json") from exc
    try:
        status = as_str(require(payload, "gate_status"), "gate_status")
        ok = as_bool(require(payload, "ok"), "ok")
        evidence = _string_tuple(payload, "evidence")
    except SchemaValidationError as exc:
        raise LAMMPSExecutionRunError(str(exc)) from exc
    if not ok or status != "worker_capability_ready":
        raise LAMMPSExecutionRunError("worker_capability_not_ready")
    return _WorkerCapabilityEvidence(
        path=path,
        gate_status=status,
        evidence=(status,) + evidence,
    )


def _manifest_payload(
    plan: _RunnablePlan,
    command: tuple[str, ...],
    execute_now: bool,
    capability: _WorkerCapabilityEvidence,
    outcome: _ProcessOutcome | None,
    missing_outputs: tuple[str, ...],
) -> JsonMap:
    return {
        "execution_result_id": f"{plan.run_id}-lammps-execution-result",
        "execution_plan_id": plan.execution_plan_id,
        "run_id": plan.run_id,
        "execution_status": _execution_status(execute_now, outcome, missing_outputs),
        "preflight_ok": True,
        "worker_capability_path": _capability_path(capability),
        "worker_capability_gate_status": capability.gate_status,
        "preflight_evidence": list(capability.evidence),
        "execute_requested": execute_now,
        "working_directory": str(plan.working_directory),
        "command": list(command),
        "command_line": _command_line(plan.working_directory, command),
        "expected_incident_count": plan.expected_incident_count,
        "required_inputs": list(plan.required_inputs),
        "expected_outputs": list(plan.expected_outputs),
        "missing_expected_outputs": list(missing_outputs),
        "return_code": _return_code(outcome),
        "stdout": _stdout(outcome),
        "stderr": _stderr(outcome),
    }


def _execution_status(
    execute_now: bool,
    outcome: _ProcessOutcome | None,
    missing_outputs: tuple[str, ...],
) -> str:
    if not execute_now:
        return "dry_run_ready"
    if outcome is None:
        return "lammps_not_started"
    if outcome.return_code != 0:
        return "lammps_failed"
    if missing_outputs:
        return "lammps_outputs_missing"
    return "lammps_completed"


def _return_code(outcome: _ProcessOutcome | None) -> int | None:
    if outcome is None:
        return None
    return outcome.return_code


def _stdout(outcome: _ProcessOutcome | None) -> str:
    if outcome is None:
        return ""
    return outcome.stdout


def _stderr(outcome: _ProcessOutcome | None) -> str:
    if outcome is None:
        return ""
    return outcome.stderr


def _command_line(run_dir: Path, command: tuple[str, ...]) -> str:
    command_text = " ".join(shlex.quote(part) for part in command)
    return f"cd {shlex.quote(str(run_dir))} && {command_text}"


def _string_tuple(payload: JsonMap, field: str) -> tuple[str, ...]:
    return tuple(as_str(item, field) for item in as_sequence(require(payload, field), field))


def _capability_path(capability: _WorkerCapabilityEvidence) -> str:
    if capability.path is None:
        return ""
    return str(capability.path)


def _optional_positive_int(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise SchemaValidationError(f"{field} must be a positive integer or null")
