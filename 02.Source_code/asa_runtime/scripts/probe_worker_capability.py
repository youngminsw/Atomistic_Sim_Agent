from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import validate_worker_capability, worker_capability_requirements_payload
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--lammps-binary", default="lmp")
    parser.add_argument("--requires-cuda", action="store_true")
    parser.add_argument("--requires-lammps", action="store_true")
    parser.add_argument("--required-lammps-packages", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = _probe_manifest(
        args.host,
        args.environment_name,
        Path(args.artifact_root),
        args.lammps_binary,
    )
    command = ("lmp",) if args.requires_lammps else ("python3",)
    requirements = worker_capability_requirements_payload(
        host_alias=args.host,
        environment_name=args.environment_name,
        remote_run_dir=args.artifact_root,
        requires_cuda=args.requires_cuda,
        command=command,
    )
    requirements = requirements | {
        "requires_lammps": args.requires_lammps,
        "required_lammps_packages": list(_csv_tuple(args.required_lammps_packages)),
    }
    report = validate_worker_capability(manifest, requirements)
    _write_json(Path(args.out), report.payload)
    print(f"worker_capability_ok={str(report.ok).lower()}")
    print(f"gate_status={report.payload['gate_status']}")
    return 0 if report.ok else 1


def _probe_manifest(
    host_alias: str,
    environment_name: str,
    artifact_root: Path,
    lammps_binary: str,
) -> JsonMap:
    lammps = _probe_lammps(lammps_binary)
    gpu = _probe_gpu()
    return {
        "host_alias": host_alias,
        "hostname": platform.node(),
        "environment_name": environment_name,
        "conda_available": shutil.which("conda") is not None,
        "conda_environment_present": _conda_env_present(environment_name),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "artifact_root": str(artifact_root),
        "artifact_root_writable": _artifact_root_writable(artifact_root),
        "gpu_available": gpu["available"],
        "gpu_model": gpu["model"],
        "cuda_visible": gpu["available"],
        "lammps_available": lammps["available"],
        "lammps_executable": lammps["executable"],
        "lammps_version": lammps["version"],
        "lammps_packages": lammps["packages"],
    }


def _conda_env_present(environment_name: str) -> bool:
    conda = shutil.which("conda")
    if conda is None:
        return False
    result = _run_command((conda, "env", "list"))
    return result.returncode == 0 and environment_name in result.stdout


def _probe_lammps(lammps_binary: str) -> JsonMap:
    executable = shutil.which(lammps_binary) or lammps_binary
    result = _run_command((executable, "-h"))
    if result.returncode != 0:
        return {"available": False, "executable": executable, "version": "", "packages": []}
    return {
        "available": True,
        "executable": executable,
        "version": _lammps_version(result.stdout),
        "packages": list(_lammps_packages(result.stdout)),
    }


def _probe_gpu() -> JsonMap:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"available": False, "model": ""}
    result = _run_command((nvidia_smi, "--query-gpu=name", "--format=csv,noheader"))
    if result.returncode != 0:
        return {"available": False, "model": ""}
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    return {"available": bool(first_line), "model": first_line}


def _artifact_root_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".worker_capability_write_test"
        marker.write_text("ok\n", encoding="utf-8")
        marker.unlink()
    except OSError:
        return False
    return True


def _run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="")


def _lammps_version(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("LAMMPS"):
            return line.strip()
    return ""


def _lammps_packages(output: str) -> tuple[str, ...]:
    for line in output.splitlines():
        normalized = line.strip()
        if normalized.lower().startswith("installed packages:"):
            return _csv_tuple(normalized.split(":", maxsplit=1)[1].replace(" ", ","))
    return ()


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
