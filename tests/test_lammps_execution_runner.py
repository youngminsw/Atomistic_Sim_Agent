from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_lammps_execution_runner_dry_run_preflights_without_lammps_binary(
    tmp_path: Path,
) -> None:
    from sim_agent.md import run_lammps_execution_plan

    _write_required_inputs(tmp_path)

    result = run_lammps_execution_plan(
        _execution_plan_payload(tmp_path, lammps_binary="missing-lmp"),
        execute_now=False,
    )

    assert result.command == ("missing-lmp", "-in", "in.atomistic_campaign")
    assert result.working_directory == tmp_path
    assert result.manifest_payload["execution_status"] == "dry_run_ready"
    assert result.manifest_payload["execute_requested"] is False
    assert result.manifest_payload["preflight_ok"] is True
    assert result.manifest_payload["command"] == ["missing-lmp", "-in", "in.atomistic_campaign"]


def test_lammps_execution_runner_execute_runs_configured_binary(tmp_path: Path) -> None:
    from sim_agent.md import run_lammps_execution_plan

    _write_required_inputs(tmp_path)
    fake_lammps = _write_fake_lammps_binary(tmp_path)

    result = run_lammps_execution_plan(
        _execution_plan_payload(tmp_path, lammps_binary=str(fake_lammps)),
        execute_now=True,
    )

    assert result.manifest_payload["execution_status"] == "lammps_completed"
    assert result.manifest_payload["execute_requested"] is True
    assert result.manifest_payload["return_code"] == 0
    assert result.manifest_payload["missing_expected_outputs"] == []
    assert "fake lammps saw -in in.atomistic_campaign" in (
        tmp_path / "log.lammps"
    ).read_text(encoding="utf-8")


def test_lammps_execution_runner_records_worker_capability_gate(tmp_path: Path) -> None:
    from sim_agent.md import run_lammps_execution_plan

    _write_required_inputs(tmp_path)
    _write_worker_capability_report(tmp_path, ok=True)

    result = run_lammps_execution_plan(
        _execution_plan_payload(tmp_path, lammps_binary="missing-lmp"),
        execute_now=False,
        worker_capability_path=tmp_path / "worker_capability.json",
    )

    assert result.manifest_payload["execution_status"] == "dry_run_ready"
    assert result.manifest_payload["worker_capability_gate_status"] == (
        "worker_capability_ready"
    )
    assert "worker_capability_ready" in result.manifest_payload["preflight_evidence"]


def test_lammps_execution_runner_rejects_failed_worker_capability(tmp_path: Path) -> None:
    from sim_agent.md import LAMMPSExecutionRunError, run_lammps_execution_plan

    _write_required_inputs(tmp_path)
    _write_worker_capability_report(tmp_path, ok=False)

    try:
        run_lammps_execution_plan(
            _execution_plan_payload(tmp_path, lammps_binary="missing-lmp"),
            execute_now=False,
            worker_capability_path=tmp_path / "worker_capability.json",
        )
    except LAMMPSExecutionRunError as exc:
        assert str(exc) == "worker_capability_not_ready"
    else:
        raise AssertionError("expected worker capability rejection")


def test_run_lammps_execution_plan_cli_writes_dry_run_manifest(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    plan_path = tmp_path / "lammps_execution_plan.json"
    out_path = tmp_path / "lammps_execution_result.json"
    plan_path.write_text(
        json.dumps(_execution_plan_payload(tmp_path, lammps_binary="missing-lmp")),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_lammps_execution_plan.py"),
            "--plan",
            str(plan_path),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "lammps_execution_runner_ok=true" in result.stdout
    assert "execution_status=dry_run_ready" in result.stdout
    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "result")
    assert payload["execution_status"] == "dry_run_ready"
    assert payload["execute_requested"] is False
    assert payload["working_directory"] == str(tmp_path)


def test_run_lammps_execution_plan_cli_records_worker_capability(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    _write_worker_capability_report(tmp_path, ok=True)
    plan_path = tmp_path / "lammps_execution_plan.json"
    out_path = tmp_path / "lammps_execution_result.json"
    plan_path.write_text(
        json.dumps(_execution_plan_payload(tmp_path, lammps_binary="missing-lmp")),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_lammps_execution_plan.py"),
            "--plan",
            str(plan_path),
            "--worker-capability",
            str(tmp_path / "worker_capability.json"),
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker_capability_gate_status=worker_capability_ready" in result.stdout
    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "result")
    assert payload["worker_capability_gate_status"] == "worker_capability_ready"


def test_run_lammps_execution_plan_cli_executes_when_requested(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    fake_lammps = _write_fake_lammps_binary(tmp_path)
    plan_path = tmp_path / "lammps_execution_plan.json"
    out_path = tmp_path / "lammps_execution_result.json"
    plan_path.write_text(
        json.dumps(_execution_plan_payload(tmp_path, lammps_binary=str(fake_lammps))),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_lammps_execution_plan.py"),
            "--plan",
            str(plan_path),
            "--out",
            str(out_path),
            "--execute",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "execution_status=lammps_completed" in result.stdout
    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "result")
    assert payload["execution_status"] == "lammps_completed"
    assert payload["return_code"] == 0


def _write_required_inputs(run_dir: Path) -> None:
    (run_dir / "in.atomistic_campaign").write_text("units metal\n", encoding="utf-8")
    (run_dir / "surface_snapshot_before.data").write_text("1 atoms\n", encoding="utf-8")
    (run_dir / "Si.tersoff").write_text("Si Si Si\n", encoding="utf-8")


def _write_fake_lammps_binary(run_dir: Path) -> Path:
    binary = run_dir / "fake_lmp.sh"
    binary.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'fake lammps saw %s %s\\n' \"$1\" \"$2\" > log.lammps\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def _write_worker_capability_report(run_dir: Path, ok: bool) -> None:
    gate_status = "worker_capability_ready" if ok else "worker_capability_rejected"
    run_dir.joinpath("worker_capability.json").write_text(
        json.dumps(
            {
                "ok": ok,
                "gate_status": gate_status,
                "evidence": ["conda_environment_present"],
                "blockers": [] if ok else ["lammps_missing"],
            }
        ),
        encoding="utf-8",
    )


def _execution_plan_payload(run_dir: Path, lammps_binary: str) -> JsonMap:
    return {
        "execution_plan_id": "run-1-execution-plan",
        "run_id": "run-1",
        "input_deck_id": "run-1-input-deck",
        "asset_manifest_id": "run-1-assets",
        "surface_state_id": "run-1-surface-state",
        "execution_status": "ready_for_lammps",
        "preflight_ok": True,
        "execute_now": False,
        "working_directory": str(run_dir),
        "lammps_binary": lammps_binary,
        "input_deck": "in.atomistic_campaign",
        "command_line": f"cd {run_dir} && {lammps_binary} -in in.atomistic_campaign",
        "required_inputs": [
            "in.atomistic_campaign",
            "surface_snapshot_before.data",
            "Si.tersoff",
        ],
        "expected_outputs": ["log.lammps"],
    }
