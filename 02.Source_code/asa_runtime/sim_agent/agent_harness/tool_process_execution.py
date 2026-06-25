from __future__ import annotations

import subprocess
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .tool_policy import DEFAULT_RUNTIME_TOOL_POLICY, is_process_allowed
from .tool_result_io import blocked_result, ledger_ref, write_result
from .tool_types import RuntimeToolCall, RuntimeToolResult


def execute_bash_process(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        argv = _argv(call.arguments)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    if not _is_allowed_process(argv):
        return blocked_result(call, session_dir, "unsafe_command", {"argv": list(argv)})
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
        return blocked_result(call, session_dir, "command_not_found", {"argv": list(argv)})
    except subprocess.TimeoutExpired:
        return blocked_result(call, session_dir, "command_timeout", {"argv": list(argv)})
    status = "succeeded" if completed.returncode == 0 else "failed"
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status=status,
            output={
                "argv": list(argv),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            artifact_ref=ledger_ref(call),
        ),
    )


def _argv(arguments: JsonMap) -> tuple[str, ...]:
    values = as_sequence(require(arguments, "argv"), "argv")
    argv = tuple(as_str(value, "argv") for value in values)
    if not argv:
        raise SchemaValidationError("argv must not be empty")
    return argv


def _is_allowed_process(argv: tuple[str, ...]) -> bool:
    return is_process_allowed(DEFAULT_RUNTIME_TOOL_POLICY, argv)
