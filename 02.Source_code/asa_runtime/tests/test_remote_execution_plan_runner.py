from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from pathlib import PurePosixPath

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import WorkerBundle, run_remote_execution_plan, worker_bundle_payload
from sim_agent.compute import remote_plan_runner


def test_render_remote_worker_plan_output_runs_through_plan_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_path = tmp_path / "worker_bundle.json"
    output_root = tmp_path / "out"
    plan_path = output_root / "remote" / "remote_plan.json"
    worker_path.write_text(json.dumps(worker_bundle_payload(_worker_bundle())), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "render_remote_worker_plan.py"),
            "--worker",
            str(worker_path),
            "--ssh-target",
            "tester@example",
            "--ssh-port",
            "2222",
            "--out",
            str(plan_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commands_seen: list[str] = []

    def successful_command(
        command: str,
        cwd: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        commands_seen.append(command)
        return subprocess.CompletedProcess(
            args=(command, cwd, timeout_s),
            returncode=0,
            stdout=f"ran={command}\n",
            stderr="",
        )

    monkeypatch.setattr(remote_plan_runner, "_run_command", successful_command)

    runner_result = run_remote_execution_plan(plan_path, timeout_s=5)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["kind"] == "remote_execution_plan"
    assert payload["created_by"] == "asa_runtime"
    assert payload["output_root"] == str(output_root.resolve())
    assert payload["plan_sha256"] == _commands_sha256(tuple(payload["all_commands"]))
    assert runner_result.ok is True
    assert commands_seen == payload["all_commands"]


def test_remote_execution_plan_runner_completes_ordered_commands(tmp_path: Path) -> None:
    plan_path = _write_remote_plan(
        tmp_path,
        (
            "echo first",
            "echo second",
        ),
    )

    result = run_remote_execution_plan(plan_path, timeout_s=5)

    assert result.ok is True
    assert result.payload["plan_status"] == "remote_plan_completed"
    assert result.payload["completed_command_count"] == 2
    assert "second" in result.payload["stdout_tail"]


def test_remote_execution_plan_runner_records_first_failure(tmp_path: Path) -> None:
    plan_path = _write_remote_plan(
        tmp_path,
        (
            "echo before",
            "echo failed >&2; exit 7",
            "echo after",
        ),
    )

    result = run_remote_execution_plan(plan_path, timeout_s=5)

    assert result.ok is False
    assert result.payload["plan_status"] == "remote_plan_failed"
    assert result.payload["returncode"] == 7
    assert result.payload["completed_command_count"] == 1
    assert result.payload["failed_command"] == "echo failed >&2; exit 7"
    assert "remote_plan_command_failed" in result.payload["blockers"]


def _write_remote_plan(tmp_path: Path, commands: tuple[str, ...]) -> Path:
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir(parents=True)
    plan_path = remote_dir / "remote_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "remote_execution_plan",
                "created_by": "asa_runtime",
                "source_root": str(SOURCE_ROOT.resolve()),
                "output_root": str(tmp_path.resolve()),
                "plan_sha256": _commands_sha256(commands),
                "ssh_target": "local-test",
                "ssh_port": 22,
                "all_commands": list(commands),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path


def _commands_sha256(commands: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(commands),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _worker_bundle() -> WorkerBundle:
    return WorkerBundle(
        host_alias="gpu-test",
        environment_name="atomistic-sim-gpu",
        run_id="plan-renderer-drift",
        remote_run_dir=PurePosixPath("/tmp/asa/plan-renderer-drift"),
        command_line="python3 run_worker.py",
        preflight_commands=("python3 --version",),
        capability_manifest_path="worker_capability.json",
        capability_requirements={},
        input_paths=(),
        output_paths=("artifacts/result.json",),
        transfer_plan=(),
        requires_cuda=False,
        uses_local_fallback=False,
    )
