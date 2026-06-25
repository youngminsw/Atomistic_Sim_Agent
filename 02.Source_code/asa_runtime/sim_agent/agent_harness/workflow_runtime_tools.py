from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_str, require
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.agents_sdk_runtime.workflow_harness import run_workflow_harness_smoke
from sim_agent.agents_sdk_runtime.workflow_runtime import respond_workflow_gate

from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult


SAFE_LEDGER_SEGMENT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True, slots=True)
class ToolBlockRequest:
    blocker: str
    output: JsonMap


def execute_workflow_start(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    if not call.caller_agent_id:
        return _blocked(
            call,
            session_dir,
            ToolBlockRequest("workflow_start_trusted_caller_required", {"tool_name": call.tool_name}),
        )
    try:
        workflow_id = as_str(require(call.arguments, "workflow_id"), "workflow_id")
        payload = _optional_mapping(call.arguments, "payload")
    except SchemaValidationError as exc:
        return _blocked(call, session_dir, ToolBlockRequest("invalid_arguments", {"error": str(exc)}))
    identity_blocker = _workflow_start_identity_blocker(call, payload)
    if identity_blocker:
        return _blocked(
            call,
            session_dir,
            ToolBlockRequest(identity_blocker, {"caller_agent_id": call.caller_agent_id}),
        )
    payload = _workflow_start_payload(call, payload)
    result = run_workflow_harness_smoke(workflow_id, payload, session_dir / "workflows")
    output: JsonMap = {
        "workflow_id": result.workflow_id,
        "status": result.status,
        "current_state": result.current_state,
        "verification_gate": result.verification_gate,
        "gate_status": result.gate_status,
        "evidence_keys": list(result.evidence_keys),
        "missing_evidence": list(result.missing_evidence),
        "resumable": result.resumable,
        "ledger_ref": f"workflows/{result.ledger_ref}",
        "blockers": list(result.blockers),
        "artifact_refs": [f"workflows/{artifact_ref}" for artifact_ref in result.artifact_refs],
        "actor_agent_id": result.actor_agent_id,
        "owner_agent_id": result.owner_agent_id,
        "target_agent_id": result.target_agent_id,
        "goal_id": result.goal_id,
    }
    if result.gate is not None:
        output["gate"] = result.gate
    blocker = result.blockers[0] if result.blockers else None
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, output, _ledger_ref(call), blocker),
    )


def execute_workflow_gate_response(call: RuntimeToolCall, session_dir: Path) -> RuntimeToolResult:
    if not call.caller_agent_id:
        return _blocked(
            call,
            session_dir,
            ToolBlockRequest("workflow_gate_trusted_caller_required", {"tool_name": call.tool_name}),
        )
    result = respond_workflow_gate(
        session_dir / "workflows",
        dict(call.arguments) | {"responder_agent_id": call.caller_agent_id},
    )
    blocker = result.blockers[0] if result.blockers else None
    output = result.to_json()
    output = dict(output) | {"ledger_ref": f"workflows/{result.ledger_ref}"}
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, result.status, output, _ledger_ref(call), blocker),
    )


def _optional_mapping(arguments: JsonMap, field: str) -> JsonMap:
    value = arguments.get(field)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise SchemaValidationError(f"{field} must be an object")


def _workflow_start_payload(call: RuntimeToolCall, payload: JsonMap) -> JsonMap:
    output = dict(payload)
    output["actor_agent_id"] = call.caller_agent_id
    output["caller_agent_id"] = call.caller_agent_id
    for field in ("owner_agent_id", "target_agent_id", "goal_id"):
        value = call.arguments.get(field)
        if isinstance(value, str) and value:
            output[field] = value
    return output


def _workflow_start_identity_blocker(call: RuntimeToolCall, payload: JsonMap) -> str:
    for source in (call.arguments, payload):
        for field in ("actor_agent_id", "caller_agent_id"):
            value = source.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or value != call.caller_agent_id:
                return "workflow_start_identity_mismatch"
    return ""


def _blocked(call: RuntimeToolCall, session_dir: Path, request: ToolBlockRequest) -> RuntimeToolResult:
    return _write_result(
        call,
        session_dir,
        RuntimeToolResult(call.tool_name, "blocked", request.output, _ledger_ref(call), request.blocker),
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


def _safe_ledger_segment(value: str, fallback: str) -> str:
    return value if SAFE_LEDGER_SEGMENT_RE.fullmatch(value) else fallback


def _ledger_ref(call: RuntimeToolCall) -> str:
    run_id = _safe_ledger_segment(call.run_id, "invalid-run-id")
    tool_name = _safe_ledger_segment(call.tool_name, "invalid-tool")
    return f"tool_ledgers/{run_id}/{tool_name}.json"
