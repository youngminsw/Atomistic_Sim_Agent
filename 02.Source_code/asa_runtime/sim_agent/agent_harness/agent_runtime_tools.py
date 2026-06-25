from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.agent_runtime import (
    HandoffTaskRequest,
    ReplyAgentMessageRequest,
    SendAgentMessageRequest,
    SubagentControlRequest,
    SubagentInspectRequest,
    SubagentTaskResult,
    SubagentTaskRequest,
    ack_agent_message,
    control_bounded_subagent,
    handoff_task,
    inspect_bounded_subagent,
    read_agent_message,
    reply_agent_message,
    run_bounded_subagent,
    send_agent_message,
)
from sim_agent.schemas._parse import JsonMap, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.agents_sdk_runtime.invocation_artifacts import skill_invocation_payload
from sim_agent.agents_sdk_runtime.skill_registry import run_registered_agent_skill

from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult


SAFE_LEDGER_SEGMENT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True, slots=True)
class ToolBlockRequest:
    blocker: str
    output: JsonMap


def execute_agent_message(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    runtime_dir = _runtime_session_dir(session_dir)
    try:
        action = _optional_str(call.arguments, "action", "send")
        match action:
            case "send":
                bus_result = send_agent_message(runtime_dir, _send_message_request(call))
            case "ack":
                bus_result = ack_agent_message(
                    runtime_dir,
                    message_id=as_str(require(call.arguments, "message_id"), "message_id"),
                    by_agent=as_str(require(call.arguments, "by_agent"), "by_agent"),
                )
            case "read":
                bus_result = read_agent_message(
                    runtime_dir,
                    message_id=as_str(require(call.arguments, "message_id"), "message_id"),
                    by_agent=as_str(require(call.arguments, "by_agent"), "by_agent"),
                )
            case "reply":
                bus_result = reply_agent_message(runtime_dir, _reply_message_request(call))
            case _:
                return _blocked(call, session_dir, ToolBlockRequest("invalid_action", {"action": action}))
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, bus_result.status, bus_result.to_json(), _ledger_ref(call), bus_result.blocker),
    )


def execute_handoff_task(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    runtime_dir = _runtime_session_dir(session_dir)
    try:
        result = handoff_task(runtime_dir, _handoff_request(call))
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, result.to_json(), _ledger_ref(call), result.blocker),
    )


def execute_subagent_task(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    runtime_dir = _runtime_session_dir(session_dir)
    try:
        result = run_bounded_subagent(runtime_dir, _subagent_task_request(call))
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, _subagent_task_output(result), _ledger_ref(call), result.blocker),
    )


def execute_subagent_inspect(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    runtime_dir = _runtime_session_dir(session_dir)
    try:
        result = inspect_bounded_subagent(runtime_dir, _subagent_inspect_request(call))
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, result.output, _ledger_ref(call), result.blocker),
    )


def execute_subagent_control(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    runtime_dir = _runtime_session_dir(session_dir)
    try:
        result = control_bounded_subagent(runtime_dir, _subagent_control_request(call))
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, result.output, _ledger_ref(call), result.blocker),
    )


def execute_skill_invoke(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    try:
        skill_id = as_str(require(call.arguments, "skill_id"), "skill_id")
        payload = _optional_mapping(call.arguments, "payload")
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    invocation = run_registered_agent_skill(skill_id, payload, session_dir)
    if invocation is None:
        return _blocked(call, session_dir, ToolBlockRequest("unknown_skill", {"skill_id": skill_id}))
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, invocation.status, skill_invocation_payload(invocation), _ledger_ref(call)),
    )


def _send_message_request(call: RuntimeToolCall) -> SendAgentMessageRequest:
    return SendAgentMessageRequest(
        from_agent=as_str(require(call.arguments, "from_agent"), "from_agent"),
        to_agent=as_str(require(call.arguments, "to_agent"), "to_agent"),
        content=as_str(require(call.arguments, "content"), "content"),
        thread_id=_optional_str(call.arguments, "thread_id", f"thread-{call.session_id}"),
        message_id=_optional_str(call.arguments, "message_id", f"msg-{call.run_id}-{call.tool_name}"),
        blocked_targets=_optional_str_tuple(call.arguments, "blocked_targets"),
    )


def _reply_message_request(call: RuntimeToolCall) -> ReplyAgentMessageRequest:
    return ReplyAgentMessageRequest(
        message_id=as_str(require(call.arguments, "message_id"), "message_id"),
        by_agent=as_str(require(call.arguments, "by_agent"), "by_agent"),
        content=as_str(require(call.arguments, "content"), "content"),
    )


def _handoff_request(call: RuntimeToolCall) -> HandoffTaskRequest:
    return HandoffTaskRequest(
        from_agent=_optional_str(call.arguments, "from_agent", "orchestrator"),
        target_agent=as_str(require(call.arguments, "target_agent"), "target_agent"),
        task_id=_optional_str(call.arguments, "task_id", f"task-{call.run_id}-{call.tool_name}"),
        thread_id=_optional_str(call.arguments, "thread_id", f"thread-{call.session_id}"),
        task=as_str(require(call.arguments, "task"), "task"),
    )


def _subagent_task_request(call: RuntimeToolCall) -> SubagentTaskRequest:
    return SubagentTaskRequest(
        caller_agent=as_str(require(call.arguments, "caller_agent"), "caller_agent"),
        preset=as_str(require(call.arguments, "preset"), "preset"),
        task_id=_optional_str(call.arguments, "task_id", f"subagent-{call.run_id}"),
        task=as_str(require(call.arguments, "task"), "task"),
        depth=_optional_int(call.arguments, "depth", 1),
    )


def _subagent_inspect_request(call: RuntimeToolCall) -> SubagentInspectRequest:
    return SubagentInspectRequest(
        caller_agent=as_str(require(call.arguments, "caller_agent"), "caller_agent"),
        preset=as_str(require(call.arguments, "preset"), "preset"),
        subagent_id=as_str(require(call.arguments, "subagent_id"), "subagent_id"),
    )


def _subagent_control_request(call: RuntimeToolCall) -> SubagentControlRequest:
    return SubagentControlRequest(
        action=as_str(require(call.arguments, "action"), "action"),
        caller_agent=as_str(require(call.arguments, "caller_agent"), "caller_agent"),
        preset=_optional_str(call.arguments, "preset", ""),
        subagent_id=_optional_str(call.arguments, "subagent_id", ""),
        content=_optional_str(call.arguments, "content", ""),
    )


def _optional_str(arguments: JsonMap, field: str, fallback: str) -> str:
    value = arguments.get(field)
    if isinstance(value, str) and value:
        return value
    return fallback


def _optional_int(arguments: JsonMap, field: str, fallback: int) -> int:
    value = arguments.get(field)
    if value is None:
        return fallback
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")


def _optional_str_tuple(arguments: JsonMap, field: str) -> tuple[str, ...]:
    value = arguments.get(field)
    if value is None:
        return ()
    values = as_sequence(value, field)
    return tuple(as_str(item, field) for item in values)


def _optional_mapping(arguments: JsonMap, field: str) -> JsonMap:
    value = arguments.get(field)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise SchemaValidationError(f"{field} must be an object")


def _runtime_session_dir(session_dir: Path) -> Path:
    if _is_global_session_dir(session_dir):
        return session_dir
    for parent in session_dir.parents:
        if _is_global_session_dir(parent):
            return parent
    return session_dir


def _is_global_session_dir(path: Path) -> bool:
    return (path / "global_session.json").is_file() and (path / "agent_sessions").is_dir()


def _blocked(call: RuntimeToolCall, session_dir: Path, request: ToolBlockRequest) -> RuntimeToolResult:
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, "blocked", request.output, _ledger_ref(call), request.blocker),
    )


def _subagent_task_output(result: SubagentTaskResult) -> JsonMap:
    return result.to_json()


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


def _safe_ledger_segment(value: str, fallback: str) -> str:
    return value if SAFE_LEDGER_SEGMENT_RE.fullmatch(value) else fallback


def _ledger_ref(call: RuntimeToolCall) -> str:
    run_id = _safe_ledger_segment(call.run_id, "invalid-run-id")
    tool_name = _safe_ledger_segment(call.tool_name, "invalid-tool")
    return f"tool_ledgers/{run_id}/{tool_name}.json"
