from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str, require

from .types import ComputePolicyError


@dataclass(frozen=True, slots=True)
class RemotePlanRunResult:
    ok: bool
    payload: JsonMap


def run_remote_execution_plan(
    plan_path: Path,
    timeout_s: float | None,
) -> RemotePlanRunResult:
    plan = _load_plan(plan_path)
    commands = _commands(plan)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    completed_count = 0
    failed_command = ""
    returncode = 0
    for command in commands:
        completed = _run_command(command, plan_path.parent, timeout_s)
        stdout_parts.append(completed.stdout)
        stderr_parts.append(completed.stderr)
        if completed.returncode != 0:
            failed_command = command
            returncode = completed.returncode
            break
        completed_count += 1
    blockers = _blockers(returncode, completed_count, len(commands))
    ok = not blockers
    return RemotePlanRunResult(
        ok=ok,
        payload={
            "ok": ok,
            "plan_status": "remote_plan_completed" if ok else "remote_plan_failed",
            "plan_path": str(plan_path),
            "returncode": returncode,
            "completed_command_count": completed_count,
            "total_command_count": len(commands),
            "failed_command": failed_command,
            "stdout_tail": _tail("\n".join(stdout_parts)),
            "stderr_tail": _tail("\n".join(stderr_parts)),
            "blockers": blockers,
        },
    )


def write_remote_execution_plan_result(output_path: Path, result: RemotePlanRunResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_plan(path: Path) -> JsonMap:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComputePolicyError(f"remote_plan_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"remote_plan_invalid_json={path}") from exc
    return as_mapping(payload, "remote_plan")


def _commands(plan: JsonMap) -> tuple[str, ...]:
    values = as_sequence(require(plan, "all_commands"), "all_commands")
    return tuple(as_str(command, "all_commands") for command in values)


def _run_command(
    command: str,
    cwd: Path,
    timeout_s: float | None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            shell=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout=stdout,
            stderr=stderr + "\nremote_plan_timeout",
        )


def _blockers(returncode: int, completed_count: int, total_count: int) -> list[str]:
    blockers: list[str] = []
    if returncode != 0:
        blockers.append("remote_plan_command_failed")
    if completed_count < total_count:
        blockers.append("remote_plan_incomplete")
    return blockers


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]
