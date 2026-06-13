from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import as_mapping


def test_agent_cli_can_run_remote_capability_probe_and_record_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "agent-cli-capability"
    env = _env_with_failing_ssh(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Probe remote capability for Ar etching on amorphous Si",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--run-remote-capability-probe",
            "--remote-run-timeout-s",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    result_path = output_dir / "remote_capability_probe_result.json"
    payload = as_mapping(
        json.loads(result_path.read_text(encoding="utf-8")),
        "remote_capability_probe_result",
    )
    assert result.returncode == 1
    assert "remote_capability_probe_result_path=" in result.stdout
    assert payload["probe_status"] == "remote_capability_failed"
    assert "remote_probe_command_failed" in payload["blockers"]
    assert "Connection reset by peer" in payload["stderr_tail"]


def test_agent_cli_can_run_amorphous_prep_plan_and_record_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "agent-cli-prep"
    env = _env_with_failing_ssh(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Run amorphous Si prep before Ar etching",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--run-amorphous-structure-prep",
            "--remote-run-timeout-s",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    result_path = output_dir / "amorphous_structure_prep_remote_result.json"
    payload = as_mapping(
        json.loads(result_path.read_text(encoding="utf-8")),
        "amorphous_structure_prep_remote_result",
    )
    ledger = as_mapping(
        json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8")),
        "agent_run_ledger",
    )
    assert result.returncode == 1
    assert "amorphous_structure_prep_remote_result_path=" in result.stdout
    assert payload["plan_status"] == "remote_plan_failed"
    assert "remote_plan_command_failed" in payload["blockers"]
    assert "Connection reset by peer" in payload["stderr_tail"]
    assert "amorphous_structure_prep_remote_result_path" in ledger["artifact_paths"]
    assert ledger["overall_status"] == "remote_failed"
    assert ledger["remote"]["amorphous_prep_status"] == "remote_plan_failed"
    assert "remote_plan_command_failed" in ledger["remote"]["amorphous_prep_blockers"]
    assert "remote_plan_command_failed" in ledger["qa"]["hard_blockers"]


def test_agent_cli_records_completed_amorphous_prep_plan(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "agent-cli-prep-ok"
    env = _env_with_successful_remote_tools(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Run amorphous Si prep before Ar etching",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--run-amorphous-structure-prep",
            "--remote-run-timeout-s",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = as_mapping(
        json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8")),
        "agent_run_ledger",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger["overall_status"] == "md_action_required"
    assert ledger["remote"]["amorphous_prep_status"] == "remote_plan_completed"
    assert ledger["remote"]["amorphous_prep_blockers"] == []
    assert "amorphous_structure_prep_remote_completed" in ledger["evidence"]


def _env_with_failing_ssh(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Connection reset by peer' >&2\n"
        "exit 255\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _env_with_successful_remote_tools(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo remote-ok\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    rsync = fake_bin / "rsync"
    rsync.write_text(
        "#!/usr/bin/env bash\n"
        "echo rsync-ok\n",
        encoding="utf-8",
    )
    rsync.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    return env
