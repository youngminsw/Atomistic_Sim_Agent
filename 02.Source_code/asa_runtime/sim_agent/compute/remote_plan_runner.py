from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .types import ComputePolicyError

_CREATED_BY = "asa_runtime"
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|password|secret|token)=([^ \t\r\n]+)|sk-[A-Za-z0-9_-]+"
)


@dataclass(frozen=True, slots=True)
class RemotePlanRunResult:
    ok: bool
    payload: JsonMap


@dataclass(frozen=True, slots=True)
class ApprovedRemotePayload:
    payload: JsonMap
    manifest_path: Path
    source_root: Path
    output_root: Path
    approved_root: Path


def run_remote_execution_plan(
    plan_path: Path,
    timeout_s: float | None,
) -> RemotePlanRunResult:
    approved = load_approved_remote_payload(plan_path, "remote_execution_plan")
    plan = approved.payload
    commands = _commands(plan)
    _verify_payload_hash(plan, "plan_sha256", _commands_sha256(commands))
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
    stdout_tail, stdout_redacted = redacted_tail("\n".join(stdout_parts))
    stderr_tail, stderr_redacted = redacted_tail("\n".join(stderr_parts))
    blockers = _blockers(
        returncode,
        completed_count,
        len(commands),
        stdout_redacted or stderr_redacted,
    )
    ok = not blockers
    return RemotePlanRunResult(
        ok=ok,
        payload={
            "ok": ok,
            "plan_status": "remote_plan_completed" if ok else "remote_plan_failed",
            "plan_path": str(approved.manifest_path),
            "returncode": returncode,
            "completed_command_count": completed_count,
            "total_command_count": len(commands),
            "failed_command": failed_command,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
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


def load_approved_remote_payload(path: Path, expected_kind: str) -> ApprovedRemotePayload:
    payload = _load_plan(path)
    resolved_path = _resolve_existing(path, "remote_manifest_unapproved_root")
    try:
        source_root = _required_root(payload, "source_root")
        output_root = _required_root(payload, "output_root")
    except ComputePolicyError as exc:
        if path.is_absolute():
            raise ComputePolicyError("remote_manifest_unapproved_root") from exc
        raise
    approved_roots = (
        (output_root / "remote").resolve(strict=False),
        (source_root / ".asa" / "remote_manifests").resolve(strict=False),
    )
    approved_root = _matching_approved_root(resolved_path, approved_roots)
    _require_provenance(payload, expected_kind)
    return ApprovedRemotePayload(
        payload=payload,
        manifest_path=resolved_path,
        source_root=source_root,
        output_root=output_root,
        approved_root=approved_root,
    )


def approved_relative_file(
    approved: ApprovedRemotePayload,
    field: str,
    blocker: str = "remote_path_escape",
) -> Path:
    raw = Path(as_str(require(approved.payload, field), field))
    if raw.is_absolute():
        raise ComputePolicyError(blocker)
    resolved = _resolve_existing(approved.approved_root / raw, blocker)
    _require_inside(resolved, approved.approved_root, blocker)
    return resolved


def approved_output_path(approved: ApprovedRemotePayload, field: str) -> Path:
    raw = Path(as_str(require(approved.payload, field), field))
    if raw.is_absolute():
        raise ComputePolicyError("remote_output_root_violation")
    candidate = (approved.approved_root / raw).resolve(strict=False)
    _require_inside(candidate, approved.output_root, "remote_output_root_violation")
    return candidate


def verify_script_hash(payload: JsonMap, script_path: Path) -> None:
    _verify_payload_hash(payload, "script_sha256", _file_sha256(script_path))


def require_normalized_bash_newlines(script_path: Path) -> None:
    data = script_path.read_bytes()
    if b"\r" in data:
        raise ComputePolicyError("remote_script_newline_not_normalized")


def redacted_tail(text: str, limit: int = 4000) -> tuple[str, bool]:
    redacted, count = _SECRET_PATTERN.subn(_redacted_secret, text)
    return (_tail(redacted, limit), count > 0)


def _commands(plan: JsonMap) -> tuple[str, ...]:
    values = as_sequence(require(plan, "all_commands"), "all_commands")
    return tuple(as_str(command, "all_commands") for command in values)


def _run_command(
    command: str,
    cwd: Path,
    timeout_s: float | None,
) -> subprocess.CompletedProcess[str]:
    shell_command = _shell_command(command)
    try:
        return subprocess.run(
            shell_command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            shell=isinstance(shell_command, str),
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


def _shell_command(command: str) -> tuple[str, str, str] | str:
    bash_path = shutil.which("bash")
    if bash_path:
        return (bash_path, "-lc", command)
    return command


def _blockers(
    returncode: int,
    completed_count: int,
    total_count: int,
    secret_redacted: bool,
) -> list[str]:
    blockers: list[str] = []
    if returncode != 0:
        blockers.append("remote_plan_command_failed")
    if completed_count < total_count:
        blockers.append("remote_plan_incomplete")
    if secret_redacted:
        blockers.append("remote_secret_tail_redacted")
    return blockers


def _required_root(payload: JsonMap, field: str) -> Path:
    try:
        raw = as_str(require(payload, field), field)
        return Path(raw).resolve(strict=True)
    except (ComputePolicyError, OSError, SchemaValidationError) as exc:
        raise ComputePolicyError("remote_manifest_missing_provenance") from exc


def _require_provenance(payload: JsonMap, expected_kind: str) -> None:
    try:
        created_by = as_str(require(payload, "created_by"), "created_by")
        kind = as_str(require(payload, "kind"), "kind")
        require(payload, "schema_version")
    except (ComputePolicyError, SchemaValidationError) as exc:
        raise ComputePolicyError("remote_manifest_missing_provenance") from exc
    if created_by != _CREATED_BY or kind != expected_kind:
        raise ComputePolicyError("remote_manifest_missing_provenance")


def _verify_payload_hash(payload: JsonMap, field: str, actual: str) -> None:
    try:
        expected = as_str(require(payload, field), field)
    except (ComputePolicyError, SchemaValidationError) as exc:
        raise ComputePolicyError("remote_manifest_missing_provenance") from exc
    if expected != actual:
        raise ComputePolicyError("remote_manifest_hash_mismatch")


def _commands_sha256(commands: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(commands),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve_existing(path: Path, blocker: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ComputePolicyError(blocker) from exc


def _matching_approved_root(path: Path, approved_roots: tuple[Path, Path]) -> Path:
    for root in approved_roots:
        if _is_relative_to(path, root):
            return root
    raise ComputePolicyError("remote_manifest_unapproved_root")


def _require_inside(path: Path, root: Path, blocker: str) -> None:
    if not _is_relative_to(path, root):
        raise ComputePolicyError(blocker)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _redacted_secret(match: re.Match[str]) -> str:
    key = match.group(1)
    if key is None:
        return "[REDACTED_SECRET]"
    return f"{key}=[REDACTED]"


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]
