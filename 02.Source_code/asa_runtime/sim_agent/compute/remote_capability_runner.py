from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .types import ComputePolicyError


@dataclass(frozen=True, slots=True)
class RemoteCapabilityProbeRunResult:
    ok: bool
    payload: JsonMap


def run_remote_capability_probe(
    manifest_path: Path,
    timeout_s: float | None,
) -> RemoteCapabilityProbeRunResult:
    manifest = _load_manifest(manifest_path)
    script_path = _resolve_script_path(manifest_path, manifest)
    completed = _run_script(script_path, timeout_s)
    output_path = _expected_output_path(script_path, manifest)
    worker_report = _load_optional_report(output_path)
    blockers = _blockers(completed, output_path, worker_report)
    ok = not blockers
    payload = {
        "ok": ok,
        "probe_status": "remote_capability_ready" if ok else "remote_capability_failed",
        "manifest_path": str(manifest_path),
        "probe_script": str(script_path),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
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


def _load_manifest(path: Path) -> JsonMap:
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")), "probe_manifest")
    except OSError as exc:
        raise ComputePolicyError(f"probe_manifest_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"probe_manifest_invalid_json={path}") from exc


def _resolve_script_path(manifest_path: Path, manifest: JsonMap) -> Path:
    raw = Path(as_str(require(manifest, "probe_script"), "probe_script"))
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    candidate = manifest_path.parent / raw
    if candidate.exists():
        return candidate.resolve()
    raise ComputePolicyError(f"probe_script_missing={raw}")


def _expected_output_path(script_path: Path, manifest: JsonMap) -> Path:
    raw = Path(as_str(require(manifest, "expected_output"), "expected_output"))
    if raw.is_absolute():
        return raw
    return script_path.parent / raw


def _run_script(script_path: Path, timeout_s: float | None) -> subprocess.CompletedProcess[str]:
    _normalize_bash_newlines(script_path)
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


def _normalize_bash_newlines(script_path: Path) -> None:
    data = script_path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized != data:
        script_path.write_bytes(normalized)


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
) -> list[str]:
    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append("remote_probe_command_failed")
    if not output_path.exists():
        blockers.append("worker_capability_output_missing")
    if worker_report and not _worker_report_ok(worker_report):
        blockers.append("worker_capability_gate_failed")
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


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]
