from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .tool_policy import DEFAULT_RUNTIME_TOOL_POLICY, is_process_allowed
from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult, ToolRegistry


SAFE_LEDGER_SEGMENT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def execute_runtime_tool(
    call: RuntimeToolCall,
    registry: ToolRegistry,
    session_dir: Path,
) -> RuntimeToolResult:
    identifier_blocker = _identifier_blocker(call)
    if identifier_blocker is not None:
        return _write_result(
            call,
            session_dir,
            RuntimeToolResult(
                tool_name=call.tool_name,
                status="blocked",
                output={"run_id": call.run_id, "tool_name": call.tool_name},
                artifact_ref=_ledger_ref(call),
                blocker=identifier_blocker,
            ),
        )
    tool = registry.require_tool(call.tool_name)
    if tool.executor is None:
        return _write_result(
            call,
            session_dir,
            RuntimeToolResult(
                tool_name=call.tool_name,
                status="blocked",
                output={"tool_name": call.tool_name},
                artifact_ref=_ledger_ref(call),
                blocker="tool_not_executable",
            ),
        )
    return tool.executor(call, session_dir)


def execute_bash_process(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        argv = _argv(call.arguments)
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, "invalid_arguments", {"error": str(exc)})
    if not _is_allowed_process(argv):
        return _blocked(call, session_dir, "unsafe_command", {"argv": list(argv)})
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=session_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError:
        return _blocked(call, session_dir, "command_not_found", {"argv": list(argv)})
    except subprocess.TimeoutExpired:
        return _blocked(call, session_dir, "command_timeout", {"argv": list(argv)})
    status = "succeeded" if completed.returncode == 0 else "failed"
    result = RuntimeToolResult(
        tool_name=call.tool_name,
        status=status,
        output={
            "argv": list(argv),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        artifact_ref=_ledger_ref(call),
    )
    return _write_result(call, session_dir, result)


def execute_artifact_write(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        relative_path = as_str(require(call.arguments, "relative_path"), "relative_path")
        content = as_str(require(call.arguments, "content"), "content")
        artifact_path = _safe_artifact_path(session_dir, relative_path)
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return _blocked(call, session_dir, exc.code, {"relative_path": call.arguments.get("relative_path")})
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(content, encoding="utf-8")
    result = RuntimeToolResult(
        tool_name=call.tool_name,
        status="succeeded",
        output={"relative_path": relative_path, "bytes_written": len(content.encode("utf-8"))},
        artifact_ref=_ledger_ref(call),
    )
    return _write_result(call, session_dir, result)


def execute_graphdb_dry_run(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        database_name = as_str(require(call.arguments, "database_name"), "database_name")
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, "invalid_arguments", {"error": str(exc)})
    result = RuntimeToolResult(
        tool_name=call.tool_name,
        status="succeeded",
        output={
            "database_name": database_name,
            "neo4j_write_enabled": False,
            "mode": "dry_run",
        },
        artifact_ref=_ledger_ref(call),
    )
    return _write_result(call, session_dir, result)


def _argv(arguments: JsonMap) -> tuple[str, ...]:
    values = as_sequence(require(arguments, "argv"), "argv")
    argv = tuple(as_str(value, "argv") for value in values)
    if not argv:
        raise SchemaValidationError("argv must not be empty")
    return argv


def _is_allowed_process(argv: tuple[str, ...]) -> bool:
    return is_process_allowed(DEFAULT_RUNTIME_TOOL_POLICY, argv)


def _safe_artifact_path(session_dir: Path, relative_path: str) -> Path:
    artifact_root = (session_dir / "artifacts").resolve()
    artifact_path = (artifact_root / relative_path).resolve()
    if artifact_path == artifact_root or artifact_root not in artifact_path.parents:
        raise RuntimeToolError("unsafe_artifact_path")
    return artifact_path


def _blocked(
    call: RuntimeToolCall,
    session_dir: Path,
    blocker: str,
    output: JsonMap,
) -> RuntimeToolResult:
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="blocked",
            output=output,
            artifact_ref=_ledger_ref(call),
            blocker=blocker,
        ),
    )


def _write_result(call: RuntimeToolCall, session_dir: Path, result: RuntimeToolResult) -> RuntimeToolResult:
    ledger_path = _safe_output_path(session_dir, result.artifact_ref)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(_result_payload(call, result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _result_payload(call: RuntimeToolCall, result: RuntimeToolResult) -> JsonMap:
    return {
        "run_id": call.run_id,
        "session_id": call.session_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "blocker": result.blocker or "",
        "output": result.output,
        "artifact_ref": result.artifact_ref,
    }


def _safe_output_path(session_dir: Path, artifact_ref: str) -> Path:
    root = session_dir.resolve()
    path = (root / artifact_ref).resolve()
    if path == root or root not in path.parents:
        raise RuntimeToolError("unsafe_ledger_path")
    return path


def _identifier_blocker(call: RuntimeToolCall) -> str | None:
    if _safe_ledger_segment(call.run_id, "invalid-run-id") != call.run_id:
        return "invalid_run_id"
    if _safe_ledger_segment(call.tool_name, "invalid-tool") != call.tool_name:
        return "invalid_tool_name"
    return None


def _safe_ledger_segment(value: str, fallback: str) -> str:
    return value if SAFE_LEDGER_SEGMENT_RE.fullmatch(value) else fallback


def _ledger_ref(call: RuntimeToolCall) -> str:
    run_id = _safe_ledger_segment(call.run_id, "invalid-run-id")
    tool_name = _safe_ledger_segment(call.tool_name, "invalid-tool")
    return f"tool_ledgers/{run_id}/{tool_name}.json"
