from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class PotentialSandboxSmokeRequest:
    candidate_payload: JsonMap
    potential_text: str
    work_dir: Path
    lammps_command: tuple[str, ...]
    timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class PotentialSandboxSmokeReport:
    ok: bool
    payload: JsonMap


def run_potential_sandbox_smoke(
    request: PotentialSandboxSmokeRequest,
) -> PotentialSandboxSmokeReport:
    if not request.lammps_command:
        return _not_run_report()
    request.work_dir.mkdir(parents=True, exist_ok=True)
    potential_path = request.work_dir / "potential_under_test.ff"
    input_path = request.work_dir / "in.potential_smoke"
    potential_path.write_text(request.potential_text, encoding="utf-8")
    input_path.write_text(_render_smoke_input(request.candidate_payload), encoding="utf-8")
    command = request.lammps_command + ("-in", str(input_path))
    try:
        result = subprocess.run(
            command,
            cwd=request.work_dir,
            text=True,
            capture_output=True,
            timeout=request.timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return _failed_report(input_path, potential_path, -1, "", "", "lammps_command_not_found")
    except subprocess.TimeoutExpired as exc:
        return _failed_report(
            input_path,
            potential_path,
            -1,
            exc.stdout or "",
            exc.stderr or "",
            "lammps_smoke_timeout",
        )
    log_text = _read_optional_text(request.work_dir / "log.lammps")
    energy_ok = _energy_smoke_passed(result.stdout, log_text)
    if result.returncode == 0 and energy_ok:
        return _passed_report(input_path, potential_path, result.stdout, result.stderr, log_text)
    return _failed_report(
        input_path,
        potential_path,
        result.returncode,
        result.stdout,
        result.stderr,
        f"lammps_return_code={result.returncode}",
    )


def _not_run_report() -> PotentialSandboxSmokeReport:
    return PotentialSandboxSmokeReport(
        ok=False,
        payload={
            "smoke_status": "sandbox_smoke_not_run",
            "syntax_smoke_passed": False,
            "energy_smoke_passed": False,
            "evidence": [],
            "errors": ["sandbox_command_required"],
        },
    )


def _passed_report(
    input_path: Path,
    potential_path: Path,
    stdout: str,
    stderr: str,
    log_text: str,
) -> PotentialSandboxSmokeReport:
    return PotentialSandboxSmokeReport(
        ok=True,
        payload={
            "smoke_status": "sandbox_smoke_passed",
            "syntax_smoke_passed": True,
            "energy_smoke_passed": True,
            "input_path": str(input_path),
            "potential_path": str(potential_path),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "log_tail": _tail(log_text),
            "evidence": ["lammps_returned_zero", "lammps_energy_smoke_passed"],
            "errors": [],
        },
    )


def _failed_report(
    input_path: Path,
    potential_path: Path,
    return_code: int,
    stdout: str,
    stderr: str,
    error: str,
) -> PotentialSandboxSmokeReport:
    return PotentialSandboxSmokeReport(
        ok=False,
        payload={
            "smoke_status": "sandbox_smoke_failed",
            "syntax_smoke_passed": False,
            "energy_smoke_passed": False,
            "input_path": str(input_path),
            "potential_path": str(potential_path),
            "return_code": return_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "evidence": [],
            "errors": [error],
        },
    )


def _render_smoke_input(candidate: JsonMap) -> str:
    unit_style = _text(candidate, "lammps_unit_style") or "metal"
    pair_style = _text(candidate, "pair_style") or "none"
    return "\n".join(
        (
            f"units {unit_style}",
            "atom_style atomic",
            "boundary p p p",
            "region box block 0 1 0 1 0 1",
            "create_box 1 box",
            "create_atoms 1 single 0.5 0.5 0.5",
            "mass 1 28.0855",
            f"pair_style {pair_style}",
            "run 0",
            "",
        )
    )


def _energy_smoke_passed(stdout: str, log_text: str) -> bool:
    combined = f"{stdout}\n{log_text}".lower()
    return "loop time" in combined or "toteng" in combined or "total energy" in combined


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _tail(value: str) -> str:
    return value[-4000:]


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""
