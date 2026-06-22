from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_run_remote_chain_cli_records_completed_stages(tmp_path: Path) -> None:
    manifest_path = _write_chain_bundle(tmp_path, script_body=_success_script())
    out_path = tmp_path / "remote_chain_result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_remote_chain.py"),
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
    assert "remote_chain_runner_ok=true" in result.stdout
    assert payload["ok"] is True
    assert payload["chain_status"] == "remote_chain_completed"
    assert payload["completed_stage_ids"] == ["01-md", "02-lammps", "03-post"]
    assert payload["missing_stage_ids"] == []


def test_run_remote_chain_cli_records_failed_stage(tmp_path: Path) -> None:
    manifest_path = _write_chain_bundle(tmp_path, script_body=_failure_script())
    out_path = tmp_path / "remote_chain_result.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_remote_chain.py"),
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
    assert payload["chain_status"] == "remote_chain_failed"
    assert payload["completed_stage_ids"] == ["01-md"]
    assert payload["missing_stage_ids"] == ["02-lammps", "03-post"]
    assert "remote_chain_command_failed" in payload["blockers"]
    assert "Connection reset by peer" in payload["stderr_tail"]


def _write_chain_bundle(tmp_path: Path, script_body: str) -> Path:
    script_path = tmp_path / "remote_chain.sh"
    manifest_path = tmp_path / "remote_chain_manifest.json"
    script_path.write_text(script_body, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "executable_script": str(script_path),
                "run_command": f"bash {script_path}",
                "stage_count": 3,
                "stage_ids": ["01-md", "02-lammps", "03-post"],
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
        "echo stage_start=01-md\n"
        "echo stage_done=01-md\n"
        "echo stage_start=02-lammps\n"
        "echo stage_done=02-lammps\n"
        "echo stage_start=03-post\n"
        "echo stage_done=03-post\n"
        "echo remote_execution_chain_done=true\n"
    )


def _failure_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "echo stage_start=01-md\n"
        "echo stage_done=01-md\n"
        "echo stage_start=02-lammps\n"
        "echo 'Connection reset by peer' >&2\n"
        "exit 255\n"
    )
