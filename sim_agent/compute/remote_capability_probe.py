from __future__ import annotations

import json
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sim_agent.schemas._parse import JsonMap

from .source_payload import SOURCE_PAYLOAD_ARCHIVE, stage_compute_source_payload
from .types import ComputePolicyError
from .worker_inventory import require_remote_worker_host


@dataclass(frozen=True, slots=True)
class RemoteCapabilityProbeBundle:
    script_path: Path
    manifest_path: Path
    source_payload_path: Path
    manifest_payload: JsonMap


def prepare_remote_capability_probe(
    source_root: Path,
    output_dir: Path,
    host_alias: str,
    environment_name: str,
    remote_user: str | None,
    ssh_target: str | None,
    ssh_port: int | None,
    requires_cuda: bool,
    requires_lammps: bool,
    required_lammps_packages: tuple[str, ...],
) -> RemoteCapabilityProbeBundle:
    host = require_remote_worker_host(host_alias, environment_name, remote_user, ssh_target, ssh_port)
    if host.ssh_port is None or host.ssh_port < 1 or host.ssh_port > 65535:
        raise ComputePolicyError(f"invalid_ssh_port={host.ssh_port}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_payload = stage_compute_source_payload(source_root, output_dir)
    remote_run_dir = _remote_run_dir(host.remote_user, host.host_alias)
    script_path = output_dir / "remote_capability_probe.sh"
    manifest_path = output_dir / "remote_capability_probe_manifest.json"
    script_path.write_text(
        _script_text(
            host.host_alias,
            host.environment_name,
            host.ssh_target,
            host.ssh_port,
            remote_run_dir,
            requires_cuda,
            requires_lammps,
            required_lammps_packages,
        ),
        encoding="utf-8",
        newline="\n",
    )
    chmod_applied = _try_set_executable(script_path)
    manifest = _manifest_payload(
        script_path,
        source_payload.archive_path,
        host.host_alias,
        host.environment_name,
        host.remote_user,
        host.ssh_target,
        host.ssh_port,
        remote_run_dir,
        host.inventory_source,
        chmod_applied,
        requires_cuda,
        requires_lammps,
        required_lammps_packages,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RemoteCapabilityProbeBundle(
        script_path=script_path,
        manifest_path=manifest_path,
        source_payload_path=source_payload.archive_path,
        manifest_payload=manifest,
    )


def _remote_run_dir(remote_user: str, host_alias: str) -> PurePosixPath:
    return (
        PurePosixPath("/home")
        / _safe_segment(remote_user, "remote_user")
        / "atomistic_sim_agent"
        / "capability"
        / _safe_segment(host_alias, "host_alias")
    )


def _script_text(
    host_alias: str,
    environment_name: str,
    ssh_target: str,
    ssh_port: int,
    remote_run_dir: PurePosixPath,
    requires_cuda: bool,
    requires_lammps: bool,
    required_lammps_packages: tuple[str, ...],
) -> str:
    remote_dir = str(remote_run_dir)
    payload_remote = f"{ssh_target}:{remote_run_dir / SOURCE_PAYLOAD_ARCHIVE}"
    extract_command = f"cd {shlex.quote(remote_dir)} && tar -xzf {SOURCE_PAYLOAD_ARCHIVE}"
    env_command = f"conda env list | grep {shlex.quote(environment_name)}"
    capability_remote = f"{ssh_target}:{remote_run_dir / 'worker_capability.json'}"
    probe_command = _probe_command(
        host_alias,
        environment_name,
        remote_run_dir,
        requires_cuda,
        requires_lammps,
        required_lammps_packages,
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "",
        _ssh(ssh_target, ssh_port, f"mkdir -p {shlex.quote(remote_dir)}"),
        _rsync(SOURCE_PAYLOAD_ARCHIVE, payload_remote, ssh_port),
        _ssh(ssh_target, ssh_port, extract_command),
        _ssh(ssh_target, ssh_port, "command -v conda"),
        _ssh(ssh_target, ssh_port, env_command),
    ]
    if requires_cuda:
        lines.append(_ssh(ssh_target, ssh_port, "nvidia-smi"))
    lines.append(_ssh(ssh_target, ssh_port, probe_command))
    lines.append(_rsync(capability_remote, "worker_capability.json", ssh_port))
    lines.append("echo 'remote_capability_probe_done=true'")
    return "\n".join(lines) + "\n"


def _probe_command(
    host_alias: str,
    environment_name: str,
    remote_run_dir: PurePosixPath,
    requires_cuda: bool,
    requires_lammps: bool,
    required_lammps_packages: tuple[str, ...],
) -> str:
    parts = [
        "python3",
        "02.Source_code/mss_agent/scripts/probe_worker_capability.py",
        "--host",
        host_alias,
        "--environment-name",
        environment_name,
        "--artifact-root",
        str(remote_run_dir),
        "--out",
        "worker_capability.json",
    ]
    if requires_cuda:
        parts.append("--requires-cuda")
    if requires_lammps:
        parts.append("--requires-lammps")
    if required_lammps_packages:
        parts.extend(("--required-lammps-packages", ",".join(required_lammps_packages)))
    return f"cd {shlex.quote(str(remote_run_dir))} && {_join(parts)}"


def _manifest_payload(
    script_path: Path,
    source_payload_path: Path,
    host_alias: str,
    environment_name: str,
    remote_user: str,
    ssh_target: str,
    ssh_port: int,
    remote_run_dir: PurePosixPath,
    inventory_source: str,
    chmod_applied: bool,
    requires_cuda: bool,
    requires_lammps: bool,
    required_lammps_packages: tuple[str, ...],
) -> JsonMap:
    return {
        "probe_script": str(script_path),
        "run_command": f"bash {script_path}",
        "source_payload_path": str(source_payload_path),
        "host_alias": host_alias,
        "environment_name": environment_name,
        "remote_user": remote_user,
        "ssh_target": ssh_target,
        "ssh_port": ssh_port,
        "remote_run_dir": str(remote_run_dir),
        "inventory_source": inventory_source,
        "expected_output": "worker_capability.json",
        "chmod_applied": chmod_applied,
        "requires_cuda": requires_cuda,
        "requires_lammps": requires_lammps,
        "required_lammps_packages": list(required_lammps_packages),
    }


def _safe_segment(raw: str, field: str) -> str:
    candidate = raw.replace("-", "").replace("_", "")
    if not candidate.isalnum():
        raise ComputePolicyError(f"unsafe_{field}={raw}")
    return raw


def _try_set_executable(script_path: Path) -> bool:
    try:
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    except OSError:
        return False
    return True


def _ssh(ssh_target: str, ssh_port: int, remote_command: str) -> str:
    return _join(("ssh", "-p", str(ssh_port), ssh_target, remote_command))


def _rsync(source: str, destination: str, ssh_port: int) -> str:
    return _join(("rsync", "-az", "-e", f"ssh -p {ssh_port}", source, destination))


def _join(parts: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)
