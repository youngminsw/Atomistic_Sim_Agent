from __future__ import annotations

from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .tool_result_io import blocked_result, ledger_ref, write_result
from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult


def execute_artifact_write(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        relative_path = as_str(require(call.arguments, "relative_path"), "relative_path")
        content = as_str(require(call.arguments, "content"), "content")
        artifact_path = _safe_artifact_path(session_dir, relative_path)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"relative_path": call.arguments.get("relative_path")})
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(content, encoding="utf-8")
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"relative_path": relative_path, "bytes_written": len(content.encode("utf-8"))},
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_file_read(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        relative_path = as_str(require(call.arguments, "relative_path"), "relative_path")
        max_bytes = _optional_positive_int(call.arguments, "max_bytes", 65536)
        file_path = _safe_workspace_path(session_dir, relative_path)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"relative_path": call.arguments.get("relative_path")})
    if not file_path.is_file():
        return blocked_result(call, session_dir, "file_not_found", {"relative_path": relative_path})
    raw = file_path.read_bytes()
    truncated = len(raw) > max_bytes
    content = raw[:max_bytes].decode("utf-8", errors="replace")
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"relative_path": relative_path, "content": content, "bytes_read": len(raw[:max_bytes]), "truncated": truncated},
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_file_write(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        relative_path = as_str(require(call.arguments, "relative_path"), "relative_path")
        content = as_str(require(call.arguments, "content"), "content")
        file_path = _safe_workspace_path(session_dir, relative_path)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"relative_path": call.arguments.get("relative_path")})
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"relative_path": relative_path, "bytes_written": len(content.encode("utf-8"))},
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_file_search(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        query = as_str(require(call.arguments, "query"), "query")
        root = _optional_str(call.arguments, "root", "")
        max_results = _optional_positive_int(call.arguments, "max_results", 20)
        search_root = _safe_workspace_search_root(session_dir, root or ".")
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"root": call.arguments.get("root")})
    if not search_root.exists():
        return blocked_result(call, session_dir, "search_root_not_found", {"root": root})
    files = [search_root] if search_root.is_file() else sorted(path for path in search_root.rglob("*") if path.is_file())
    workspace_root = _workspace_root(session_dir)
    matches: list[JsonMap] = []
    for file_path in files:
        if len(matches) >= max_results:
            break
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if query not in line:
                continue
            matches.append({"relative_path": str(file_path.relative_to(workspace_root)), "line": line_number, "preview": line[:240]})
            if len(matches) >= max_results:
                break
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"query": query, "root": root, "matches": matches, "match_count": len(matches)},
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_file_edit(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        relative_path = as_str(require(call.arguments, "relative_path"), "relative_path")
        search = as_str(require(call.arguments, "search"), "search")
        replace = as_str(require(call.arguments, "replace"), "replace")
        expected_replacements = _optional_int(call.arguments, "expected_replacements")
        file_path = _safe_workspace_path(session_dir, relative_path)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"relative_path": call.arguments.get("relative_path")})
    if not file_path.is_file():
        return blocked_result(call, session_dir, "file_not_found", {"relative_path": relative_path})
    content = file_path.read_text(encoding="utf-8")
    replacement_count = content.count(search)
    if replacement_count == 0:
        return blocked_result(call, session_dir, "search_not_found", {"relative_path": relative_path})
    if expected_replacements is not None and replacement_count != expected_replacements:
        return blocked_result(call, session_dir, "replacement_count_mismatch", {"relative_path": relative_path, "actual": replacement_count, "expected": expected_replacements})
    file_path.write_text(content.replace(search, replace), encoding="utf-8")
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"relative_path": relative_path, "replacements": replacement_count},
            artifact_ref=ledger_ref(call),
        ),
    )


def _workspace_root(session_dir: Path) -> Path:
    return (session_dir / "workspace").resolve()


def _safe_workspace_path(session_dir: Path, relative_path: str) -> Path:
    workspace_root = _workspace_root(session_dir)
    file_path = (workspace_root / relative_path).resolve()
    if file_path == workspace_root or workspace_root not in file_path.parents:
        raise RuntimeToolError("unsafe_file_path")
    return file_path


def _safe_workspace_search_root(session_dir: Path, relative_path: str) -> Path:
    workspace_root = _workspace_root(session_dir)
    search_root = (workspace_root / relative_path).resolve()
    if search_root != workspace_root and workspace_root not in search_root.parents:
        raise RuntimeToolError("unsafe_file_path")
    return search_root


def _safe_artifact_path(session_dir: Path, relative_path: str) -> Path:
    artifact_root = (session_dir / "artifacts").resolve()
    artifact_path = (artifact_root / relative_path).resolve()
    if artifact_path == artifact_root or artifact_root not in artifact_path.parents:
        raise RuntimeToolError("unsafe_artifact_path")
    return artifact_path


def _optional_str(arguments: JsonMap, field: str, fallback: str) -> str:
    value = arguments.get(field)
    if value is None:
        return fallback
    return as_str(value, field)


def _optional_int(arguments: JsonMap, field: str) -> int | None:
    value = arguments.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")


def _optional_positive_int(arguments: JsonMap, field: str, fallback: int) -> int:
    value = _optional_int(arguments, field)
    if value is None:
        return fallback
    if value <= 0:
        raise SchemaValidationError(f"{field} must be positive")
    return value
