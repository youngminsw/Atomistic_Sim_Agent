from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str, require

from .types import ComputePolicyError


@dataclass(frozen=True, slots=True)
class RemoteChainRunResult:
    ok: bool
    payload: JsonMap


def run_remote_chain(manifest_path: Path, timeout_s: float | None) -> RemoteChainRunResult:
    manifest = _load_manifest(manifest_path)
    script_path = _resolve_script_path(manifest_path, manifest)
    stage_ids = _stage_ids(manifest)
    completed = _run_script(script_path, timeout_s)
    completed_stages = _completed_stage_ids(completed.stdout, stage_ids)
    missing_stages = tuple(stage for stage in stage_ids if stage not in completed_stages)
    blockers = _blockers(completed, missing_stages)
    ok = not blockers
    payload = {
        "ok": ok,
        "chain_status": "remote_chain_completed" if ok else "remote_chain_failed",
        "manifest_path": str(manifest_path),
        "chain_script": str(script_path),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "stage_ids": list(stage_ids),
        "completed_stage_ids": list(completed_stages),
        "missing_stage_ids": list(missing_stages),
        "blockers": blockers,
    }
    return RemoteChainRunResult(ok=ok, payload=payload)


def write_remote_chain_result(output_path: Path, result: RemoteChainRunResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> JsonMap:
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")), "remote_chain_manifest")
    except OSError as exc:
        raise ComputePolicyError(f"remote_chain_manifest_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"remote_chain_manifest_invalid_json={path}") from exc


def _resolve_script_path(manifest_path: Path, manifest: JsonMap) -> Path:
    raw = Path(as_str(require(manifest, "executable_script"), "executable_script"))
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    candidate = manifest_path.parent / raw
    if candidate.exists():
        return candidate.resolve()
    raise ComputePolicyError(f"remote_chain_script_missing={raw}")


def _stage_ids(manifest: JsonMap) -> tuple[str, ...]:
    values = as_sequence(require(manifest, "stage_ids"), "stage_ids")
    return tuple(as_str(item, "stage_ids") for item in values)


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
            stderr=stderr + "\nremote_chain_timeout",
        )


def _normalize_bash_newlines(script_path: Path) -> None:
    data = script_path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized != data:
        script_path.write_bytes(normalized)


def _completed_stage_ids(stdout: str, stage_ids: tuple[str, ...]) -> tuple[str, ...]:
    completed: list[str] = []
    valid = frozenset(stage_ids)
    for line in stdout.splitlines():
        if line.startswith("stage_done="):
            stage_id = line.split("=", maxsplit=1)[1].strip("'\"")
            if stage_id in valid and stage_id not in completed:
                completed.append(stage_id)
    return tuple(completed)


def _blockers(
    completed: subprocess.CompletedProcess[str],
    missing_stages: tuple[str, ...],
) -> list[str]:
    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append("remote_chain_command_failed")
    if missing_stages:
        blockers.append("remote_chain_stage_incomplete")
    if completed.returncode == 0 and "remote_execution_chain_done=true" not in completed.stdout:
        blockers.append("remote_chain_completion_marker_missing")
    return blockers


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]
