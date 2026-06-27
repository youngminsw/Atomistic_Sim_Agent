from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, require
from sim_agent.schemas.errors import SchemaValidationError

from .remote_plan_runner import (
    approved_output_path,
    approved_relative_file,
    load_approved_remote_payload,
    redacted_tail,
    require_normalized_bash_newlines,
    verify_script_hash,
)


@dataclass(frozen=True, slots=True)
class RemoteCapabilityProbeRunResult:
    ok: bool
    payload: JsonMap


def run_remote_capability_probe(
    manifest_path: Path,
    timeout_s: float | None,
) -> RemoteCapabilityProbeRunResult:
    approved = load_approved_remote_payload(manifest_path, "remote_capability_probe")
    manifest = approved.payload
    script_path = approved_relative_file(approved, "probe_script")
    verify_script_hash(manifest, script_path)
    require_normalized_bash_newlines(script_path)
    output_path = approved_output_path(approved, "expected_output")
    completed = _run_script(script_path, timeout_s)
    worker_report = _load_optional_report(output_path)
    stdout_tail, stdout_redacted = redacted_tail(completed.stdout)
    stderr_tail, stderr_redacted = redacted_tail(completed.stderr)
    blockers = _blockers(
        completed,
        output_path,
        worker_report,
        stdout_redacted or stderr_redacted,
    )
    ok = not blockers
    payload = {
        "ok": ok,
        "probe_status": "remote_capability_ready" if ok else "remote_capability_failed",
        "manifest_path": str(approved.manifest_path),
        "probe_script": str(script_path),
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "expected_output": str(output_path),
        "expected_output_exists": output_path.exists(),
        "worker_capability_gate_status": _gate_status(worker_report),
        "worker_capability_report": worker_report,
        "blockers": blockers,
    }
    return RemoteCapabilityProbeRunResult(ok=ok, payload=payload)


def write_remote_capability_probe_result(
    output_path: Path,
    result: RemoteCapabilityProbeRunResult,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_script(script_path: Path, timeout_s: float | None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("bash", script_path.name),
            cwd=script_path.parent,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            ("bash", str(script_path)),
            returncode=124,
            stdout=stdout,
            stderr=stderr + "\nremote_probe_timeout",
        )

def _load_optional_report(path: Path) -> JsonMap:
    if not path.exists():
        return {}
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")), "worker_capability")
    except (OSError, json.JSONDecodeError, SchemaValidationError):
        return {"gate_status": "worker_capability_unreadable", "ok": False}


def _blockers(
    completed: subprocess.CompletedProcess[str],
    output_path: Path,
    worker_report: JsonMap,
    secret_redacted: bool,
) -> list[str]:
    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append("remote_probe_command_failed")
    if not output_path.exists():
        blockers.append("worker_capability_output_missing")
    if worker_report and not _worker_report_ok(worker_report):
        blockers.append("worker_capability_gate_failed")
    if secret_redacted:
        blockers.append("remote_secret_tail_redacted")
    return blockers


def _worker_report_ok(report: JsonMap) -> bool:
    value = report.get("ok")
    if isinstance(value, bool):
        return value
    try:
        return as_bool(require(report, "ok"), "ok")
    except SchemaValidationError:
        return False


def _gate_status(report: JsonMap) -> str:
    value = report.get("gate_status")
    if isinstance(value, str):
        return value
    return ""
