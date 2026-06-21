from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_prepare_remote_capability_probe_cli_writes_probe_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "remote-probe"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "prepare_remote_capability_probe.py"),
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--remote-user",
            "swym",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--requires-cuda",
            "--requires-lammps",
            "--required-lammps-packages",
            "MANYBODY",
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    script_path = out_dir / "remote_capability_probe.sh"
    manifest_path = out_dir / "remote_capability_probe_manifest.json"
    payload_path = out_dir / "source_payload.tar.gz"
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote_capability_probe_ok=true" in result.stdout
    assert f"probe_script_path={script_path}" in result.stdout
    assert script_path.exists()
    assert manifest_path.exists()
    assert payload_path.exists()
    script = script_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "tar -xzf source_payload.tar.gz" in script
    assert "probe_worker_capability.py" in script
    assert "--requires-lammps" in script
    assert "--required-lammps-packages MANYBODY" in script
    assert "worker_capability.json" in script
    assert manifest["ssh_target"] == "swym@10.24.12.85"
    assert manifest["host_alias"] == "gpu-5090"
    assert manifest["run_command"] == f"bash {script_path}"
    assert manifest["expected_output"] == "worker_capability.json"
    with tarfile.open(payload_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "02.Source_code/asa_runtime/scripts/probe_worker_capability.py" in names


def test_prepare_remote_capability_probe_cli_uses_5090_inventory_default(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "remote-probe-default"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "prepare_remote_capability_probe.py"),
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--requires-cuda",
            "--requires-lammps",
            "--required-lammps-packages",
            "MANYBODY",
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = json.loads((out_dir / "remote_capability_probe_manifest.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest["ssh_target"] == "swym@10.24.12.85"
    assert manifest["ssh_port"] == 55555
    assert manifest["remote_user"] == "swym"
    assert manifest["inventory_source"] == "runtime_config"
