from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap, as_bool, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class WorkerCapabilityReport:
    ok: bool
    payload: JsonMap


def worker_capability_requirements_payload(
    host_alias: str,
    environment_name: str,
    remote_run_dir: str,
    requires_cuda: bool,
    command: tuple[str, ...],
) -> JsonMap:
    requires_lammps = _requires_lammps(command)
    return {
        "host_alias": host_alias,
        "environment_name": environment_name,
        "artifact_root": remote_run_dir,
        "requires_cuda": requires_cuda,
        "requires_lammps": requires_lammps,
        "required_lammps_packages": ["MANYBODY"] if requires_lammps else [],
    }


def validate_worker_capability(
    manifest_payload: JsonMap,
    requirements_payload: JsonMap,
) -> WorkerCapabilityReport:
    blockers: list[str] = []
    evidence: list[str] = []
    _record_identity(manifest_payload, requirements_payload, blockers, evidence)
    _record_python_conda(manifest_payload, blockers, evidence)
    _record_artifact_root(manifest_payload, blockers, evidence)
    if _required_bool(requirements_payload, "requires_cuda", blockers):
        _record_gpu(manifest_payload, blockers, evidence)
    if _required_bool(requirements_payload, "requires_lammps", blockers):
        _record_lammps(manifest_payload, requirements_payload, blockers, evidence)

    ok = not blockers
    return WorkerCapabilityReport(
        ok=ok,
        payload={
            "ok": ok,
            "gate_status": "worker_capability_ready" if ok else "worker_capability_rejected",
            "requirements": requirements_payload,
            "capability_manifest": manifest_payload,
            "evidence": evidence,
            "blockers": blockers,
        },
    )


def _record_identity(
    manifest: JsonMap,
    requirements: JsonMap,
    blockers: list[str],
    evidence: list[str],
) -> None:
    if _text(manifest, "host_alias") == _text(requirements, "host_alias"):
        evidence.append("host_alias_matches")
    else:
        blockers.append("host_alias_mismatch")
    if _text(manifest, "environment_name") == _text(requirements, "environment_name"):
        evidence.append("environment_name_matches")
    else:
        blockers.append("environment_name_mismatch")


def _record_python_conda(manifest: JsonMap, blockers: list[str], evidence: list[str]) -> None:
    _record_required_true(manifest, "conda_available", "conda_unavailable", blockers, evidence)
    _record_required_true(
        manifest,
        "conda_environment_present",
        "conda_environment_missing",
        blockers,
        evidence,
    )
    if _text(manifest, "python_executable") and _text(manifest, "python_version"):
        evidence.append("python_runtime_present")
    else:
        blockers.append("python_runtime_missing")


def _record_artifact_root(manifest: JsonMap, blockers: list[str], evidence: list[str]) -> None:
    if _text(manifest, "artifact_root"):
        evidence.append("artifact_root_present")
    else:
        blockers.append("artifact_root_missing")
    _record_required_true(
        manifest,
        "artifact_root_writable",
        "artifact_root_not_writable",
        blockers,
        evidence,
    )


def _record_gpu(manifest: JsonMap, blockers: list[str], evidence: list[str]) -> None:
    if (
        _bool(manifest, "gpu_available")
        and _bool(manifest, "cuda_visible")
        and _text(manifest, "gpu_model")
    ):
        evidence.append("gpu_capability_present")
    else:
        blockers.append("gpu_capability_missing")


def _record_lammps(
    manifest: JsonMap,
    requirements: JsonMap,
    blockers: list[str],
    evidence: list[str],
) -> None:
    _record_required_true(manifest, "lammps_available", "lammps_missing", blockers, evidence)
    if _text(manifest, "lammps_executable") and _text(manifest, "lammps_version"):
        evidence.append("lammps_runtime_present")
    else:
        blockers.append("lammps_runtime_missing")
    packages = frozenset(_str_sequence(manifest, "lammps_packages"))
    missing = tuple(
        package
        for package in _str_sequence(requirements, "required_lammps_packages")
        if package not in packages
    )
    if missing:
        blockers.extend(f"lammps_required_package_missing:{package}" for package in missing)
    else:
        evidence.append("lammps_required_packages_present")


def _record_required_true(
    manifest: JsonMap,
    field: str,
    blocker: str,
    blockers: list[str],
    evidence: list[str],
) -> None:
    if _bool(manifest, field):
        evidence.append(f"{field}")
    else:
        blockers.append(blocker)


def _requires_lammps(command: tuple[str, ...]) -> bool:
    for part in command:
        if part in ("lmp", "lammps"):
            return True
        if part.endswith("run_lammps_execution_plan.py"):
            return True
    return False


def _required_bool(payload: JsonMap, field: str, blockers: list[str]) -> bool:
    try:
        return as_bool(require(payload, field), field)
    except SchemaValidationError:
        blockers.append(f"{field}_missing")
        return False


def _bool(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, bool) and value


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def _str_sequence(payload: JsonMap, field: str) -> tuple[str, ...]:
    try:
        values = as_sequence(require(payload, field), field)
    except SchemaValidationError:
        return ()
    parsed: list[str] = []
    for value in values:
        try:
            parsed.append(as_str(value, field))
        except SchemaValidationError:
            continue
    return tuple(parsed)
