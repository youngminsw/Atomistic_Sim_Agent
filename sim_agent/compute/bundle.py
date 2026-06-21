from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .capability import worker_capability_requirements_payload
from .policy import compute_resource_for_host, require_allowed_host
from .types import ComputePolicyError, JobBundleSpec, WorkerBundle


def load_job_bundle(path: Path) -> JobBundleSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComputePolicyError(f"job_bundle_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"job_bundle_invalid_json={path}") from exc
    try:
        return _job_from_mapping(as_mapping(payload, "job_bundle"))
    except SchemaValidationError as exc:
        raise ComputePolicyError(str(exc)) from exc


def build_worker_bundle(host_alias: str, job: JobBundleSpec, remote_user: str) -> WorkerBundle:
    target = require_allowed_host(host_alias)
    resource = compute_resource_for_host(host_alias)
    if job.requires_cuda and "gpu" not in resource.roles:
        raise ComputePolicyError(f"host_role_not_allowed={host_alias}:requires_gpu")
    run_id = _safe_segment(job.job_id, "job_id")
    user_segment = _safe_segment(remote_user, "remote_user")
    remote_run_dir = PurePosixPath("/home") / user_segment / "atomistic_sim_agent" / "runs" / run_id
    capability_path = "worker_capability.json"
    capability_requirements = worker_capability_requirements_payload(
        host_alias=target.host_alias,
        environment_name=job.environment_name,
        remote_run_dir=str(remote_run_dir),
        requires_cuda=job.requires_cuda,
        command=job.command,
    )
    command_line = _command_line(remote_run_dir, job.environment_name, job.command)
    return WorkerBundle(
        host_alias=target.host_alias,
        environment_name=job.environment_name,
        run_id=run_id,
        remote_run_dir=remote_run_dir,
        command_line=command_line,
        preflight_commands=_preflight_commands(
            target.host_alias,
            job.environment_name,
            job.requires_cuda,
            remote_run_dir,
            capability_path,
            capability_requirements,
        ),
        capability_manifest_path=capability_path,
        capability_requirements=capability_requirements,
        input_paths=job.input_paths,
        output_paths=job.output_paths,
        transfer_plan=_transfer_plan(job.input_paths, job.output_paths, remote_run_dir),
        requires_cuda=job.requires_cuda,
        uses_local_fallback=target.uses_local_fallback,
    )


def worker_bundle_payload(bundle: WorkerBundle) -> JsonMap:
    return {
        "host_alias": bundle.host_alias,
        "environment_name": bundle.environment_name,
        "run_id": bundle.run_id,
        "remote_run_dir": str(bundle.remote_run_dir),
        "command_line": bundle.command_line,
        "preflight_commands": list(bundle.preflight_commands),
        "capability_manifest_path": bundle.capability_manifest_path,
        "capability_requirements": bundle.capability_requirements,
        "input_paths": list(bundle.input_paths),
        "output_paths": list(bundle.output_paths),
        "transfer_plan": list(bundle.transfer_plan),
        "requires_cuda": bundle.requires_cuda,
        "uses_local_fallback": bundle.uses_local_fallback,
    }


def job_bundle_payload(job: JobBundleSpec) -> JsonMap:
    return {
        "job_id": job.job_id,
        "environment_name": job.environment_name,
        "command": list(job.command),
        "inputs": list(job.input_paths),
        "outputs": list(job.output_paths),
        "requires_cuda": job.requires_cuda,
    }


def _job_from_mapping(value: JsonMap) -> JobBundleSpec:
    return JobBundleSpec(
        job_id=as_str(require(value, "job_id"), "job_id"),
        environment_name=as_str(require(value, "environment_name"), "environment_name"),
        command=_str_tuple(value, "command"),
        input_paths=_str_tuple(value, "inputs"),
        output_paths=_str_tuple(value, "outputs"),
        requires_cuda=as_bool(value.get("requires_cuda", False), "requires_cuda"),
    )


def _str_tuple(value: JsonMap, field: str) -> tuple[str, ...]:
    return tuple(as_str(item, field) for item in as_sequence(require(value, field), field))


def _safe_segment(raw: str, field: str) -> str:
    candidate = raw.replace("-", "").replace("_", "")
    if not candidate.isalnum():
        raise ComputePolicyError(f"unsafe_{field}={raw}")
    return raw


def _command_line(
    remote_run_dir: PurePosixPath,
    environment_name: str,
    command: tuple[str, ...],
) -> str:
    run_dir = shlex.quote(str(remote_run_dir))
    env_name = shlex.quote(environment_name)
    command_text = " ".join(shlex.quote(part) for part in command)
    return f"cd {run_dir} && conda run -n {env_name} {command_text}"


def _preflight_commands(
    host_alias: str,
    environment_name: str,
    requires_cuda: bool,
    remote_run_dir: PurePosixPath,
    capability_path: str,
    requirements: JsonMap,
) -> tuple[str, ...]:
    commands = (
        "command -v conda",
        f"conda env list | grep {shlex.quote(environment_name)}",
    )
    if requires_cuda:
        commands = commands + ("nvidia-smi",)
    return commands + (
        _capability_probe_command(
            host_alias,
            environment_name,
            remote_run_dir,
            capability_path,
            requirements,
        ),
    )


def _capability_probe_command(
    host_alias: str,
    environment_name: str,
    remote_run_dir: PurePosixPath,
    capability_path: str,
    requirements: JsonMap,
) -> str:
    parts = [
        "python3",
        "02.Source_code/asa_runtime/scripts/probe_worker_capability.py",
        "--host",
        host_alias,
        "--environment-name",
        environment_name,
        "--artifact-root",
        str(remote_run_dir),
        "--out",
        capability_path,
    ]
    if _bool_field(requirements, "requires_cuda"):
        parts.append("--requires-cuda")
    if _bool_field(requirements, "requires_lammps"):
        parts.append("--requires-lammps")
        packages = _package_csv(requirements)
        if packages:
            parts.extend(("--required-lammps-packages", packages))
    return f"cd {shlex.quote(str(remote_run_dir))} && {_shell_join(tuple(parts))}"


def _shell_join(parts: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _bool_field(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, bool) and value


def _package_csv(payload: JsonMap) -> str:
    value = payload.get("required_lammps_packages")
    if not isinstance(value, list | tuple):
        return ""
    return ",".join(str(item) for item in value if isinstance(item, str) and item)


def _transfer_plan(
    input_paths: tuple[str, ...],
    output_paths: tuple[str, ...],
    remote_run_dir: PurePosixPath,
) -> tuple[str, ...]:
    input_steps = tuple(f"upload:{path}->{remote_run_dir / path}" for path in input_paths)
    output_steps = tuple(f"download:{remote_run_dir / path}->{path}" for path in output_paths)
    return input_steps + output_steps
