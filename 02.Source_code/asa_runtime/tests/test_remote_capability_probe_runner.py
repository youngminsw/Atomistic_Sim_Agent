from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_run_remote_capability_probe_cli_records_ready_result(tmp_path: Path) -> None:
    manifest_path = _write_probe_bundle(tmp_path, script_body=_success_script())
    out_path = tmp_path / "remote_capability_probe_result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_remote_capability_probe.py"),
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote_capability_probe_runner_ok=true" in result.stdout
    assert payload["ok"] is True
    assert payload["probe_status"] == "remote_capability_ready"
    assert payload["expected_output_exists"] is True
    assert payload["worker_capability_gate_status"] == "worker_capability_ready"


def test_run_remote_capability_probe_cli_records_failed_result(tmp_path: Path) -> None:
    manifest_path = _write_probe_bundle(tmp_path, script_body=_failure_script())
    out_path = tmp_path / "remote_capability_probe_result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_remote_capability_probe.py"),
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["probe_status"] == "remote_capability_failed"
    assert "remote_probe_command_failed" in payload["blockers"]
    assert "Connection reset by peer" in payload["stderr_tail"]


def test_run_remote_capability_probe_accepts_cwd_relative_script_path(
    tmp_path: Path,
) -> None:
    manifest_path = _write_probe_bundle(tmp_path, script_body=_success_script())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["probe_script"] = "remote_capability_probe.sh"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_path = tmp_path / "relative_result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_remote_capability_probe.py"),
            "--manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        cwd=tmp_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["ok"] is True
    assert Path(payload["probe_script"]).is_absolute()


def _write_probe_bundle(tmp_path: Path, script_body: str) -> Path:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir(parents=True)
    script_path = remote_dir / "remote_capability_probe.sh"
    manifest_path = remote_dir / "remote_capability_probe_manifest.json"
    script_path.write_text(script_body, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "remote_capability_probe",
                "created_by": "asa_runtime",
                "source_root": str(SOURCE_ROOT.resolve()),
                "output_root": str(tmp_path.resolve()),
                "probe_script": script_path.name,
                "script_sha256": sha256(script_path.read_bytes()).hexdigest(),
                "run_command": f"bash {script_path}",
                "expected_output": "worker_capability.json",
                "host_alias": "gpu-5090",
                "environment_name": "atomistic-sim-gpu",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _success_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cat > worker_capability.json <<'JSON'\n"
        '{"ok": true, "gate_status": "worker_capability_ready"}\n'
        "JSON\n"
        "echo remote_capability_probe_done=true\n"
    )


def _failure_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "echo 'Connection reset by peer' >&2\n"
        "exit 255\n"
    )
