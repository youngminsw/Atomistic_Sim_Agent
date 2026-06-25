from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.knowledge.mcp_manager import call_mcp_tool, list_mcp_tools, mcp_call_tool_payload, mcp_list_tools_payload

from .tool_result_io import blocked_result, ledger_ref, safe_ledger_segment, write_result
from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult


def execute_graphdb_dry_run(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        database_name = as_str(require(call.arguments, "database_name"), "database_name")
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"database_name": database_name, "neo4j_write_enabled": False, "mode": "dry_run"},
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_mcp_list_tools(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        server_name = _optional_str(call.arguments, "server_name", "")
        result = list_mcp_tools(server_name)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    output = mcp_list_tools_payload(result)
    if result.status == "blocked":
        return blocked_result(call, session_dir, result.blocker, output)
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output=output,
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_mcp_call_tool(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        server_name = as_str(require(call.arguments, "server_name"), "server_name")
        tool_name = as_str(require(call.arguments, "tool_name"), "tool_name")
        arguments = as_mapping(call.arguments.get("arguments", {}), "arguments")
        result = call_mcp_tool(server_name, tool_name, arguments)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    output = mcp_call_tool_payload(result)
    if result.status == "blocked":
        return blocked_result(call, session_dir, result.blocker, output)
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output=output,
            artifact_ref=ledger_ref(call),
        ),
    )


def execute_custom_tool_register(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        name = as_str(require(call.arguments, "name"), "name")
        description = as_str(require(call.arguments, "description"), "description")
        parameters = as_mapping(require(call.arguments, "parameters"), "parameters")
        _validate_custom_tool_schema(name, parameters)
        descriptor_path = _safe_custom_tool_path(session_dir, name)
    except SchemaValidationError as exc:
        return blocked_result(call, session_dir, "invalid_arguments", {"error": str(exc)})
    except RuntimeToolError as exc:
        return blocked_result(call, session_dir, exc.code, {"name": call.arguments.get("name")})
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "name": name,
        "description": description,
        "parameters": dict(parameters),
        "mode": "registered_descriptor",
        "executable": False,
    }
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="succeeded",
            output={"name": name, "descriptor_ref": str(descriptor_path.relative_to(session_dir.resolve()))},
            artifact_ref=ledger_ref(call),
        ),
    )


def _safe_custom_tool_path(session_dir: Path, name: str) -> Path:
    if safe_ledger_segment(name, "invalid-custom-tool") != name:
        raise RuntimeToolError("invalid_custom_tool_name")
    return (session_dir.resolve() / "custom_tools" / f"{name}.json").resolve()


def _validate_custom_tool_schema(name: str, parameters: JsonMap) -> None:
    if safe_ledger_segment(name, "invalid-custom-tool") != name:
        raise RuntimeToolError("invalid_custom_tool_name")
    if parameters.get("type") != "object":
        raise RuntimeToolError("invalid_custom_tool_schema")
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise RuntimeToolError("invalid_custom_tool_schema")
    required = parameters.get("required", [])
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise RuntimeToolError("invalid_custom_tool_schema")
    if any(item not in properties for item in required):
        raise RuntimeToolError("invalid_custom_tool_schema")


def _optional_str(arguments: JsonMap, field: str, fallback: str) -> str:
    value = arguments.get(field)
    if value is None:
        return fallback
    return as_str(value, field)
