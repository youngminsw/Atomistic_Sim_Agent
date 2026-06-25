from __future__ import annotations

from pathlib import Path

from .tool_descriptor_execution import execute_custom_tool_register, execute_graphdb_dry_run, execute_mcp_call_tool, execute_mcp_list_tools
from .tool_process_execution import execute_bash_process
from .tool_result_io import blocked_result, identifier_blocker, ledger_ref, write_result
from .tool_session_file_execution import (
    execute_artifact_write,
    execute_file_edit,
    execute_file_read,
    execute_file_search,
    execute_file_write,
)
from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult, ToolRegistry


def execute_runtime_tool(
    call: RuntimeToolCall,
    registry: ToolRegistry,
    session_dir: Path,
) -> RuntimeToolResult:
    blocked_identifier = identifier_blocker(call)
    if blocked_identifier is not None:
        return write_result(
            call,
            session_dir,
            RuntimeToolResult(
                tool_name=call.tool_name,
                status="blocked",
                output={"run_id": call.run_id, "tool_name": call.tool_name},
                artifact_ref=ledger_ref(call),
                blocker=blocked_identifier,
            ),
        )
    try:
        tool = registry.require_tool(call.tool_name)
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"tool_name": call.tool_name})
    if tool.executor is None:
        return blocked_result(call, session_dir, "tool_not_executable", {"tool_name": call.tool_name})
    return tool.executor(call, session_dir)


__all__ = [
    "execute_artifact_write",
    "execute_bash_process",
    "execute_custom_tool_register",
    "execute_file_edit",
    "execute_file_read",
    "execute_file_search",
    "execute_file_write",
    "execute_graphdb_dry_run",
    "execute_mcp_call_tool",
    "execute_mcp_list_tools",
    "execute_runtime_tool",
]
