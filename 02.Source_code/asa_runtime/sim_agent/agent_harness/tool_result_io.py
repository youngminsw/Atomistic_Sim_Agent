from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolResult


SAFE_LEDGER_SEGMENT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def blocked_result(
    call: RuntimeToolCall,
    session_dir: Path,
    blocker: str,
    output: JsonMap,
) -> RuntimeToolResult:
    return write_result(
        call,
        session_dir,
        RuntimeToolResult(
            tool_name=call.tool_name,
            status="blocked",
            output=output,
            artifact_ref=ledger_ref(call),
            blocker=blocker,
        ),
    )


def write_result(call: RuntimeToolCall, session_dir: Path, result: RuntimeToolResult) -> RuntimeToolResult:
    ledger_path = safe_output_path(session_dir, result.artifact_ref)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(result_payload(call, result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def result_payload(call: RuntimeToolCall, result: RuntimeToolResult) -> JsonMap:
    return {
        "run_id": call.run_id,
        "session_id": call.session_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "blocker": result.blocker or "",
        "output": result.output,
        "artifact_ref": result.artifact_ref,
    }


def safe_output_path(session_dir: Path, artifact_ref: str) -> Path:
    root = session_dir.resolve()
    path = (root / artifact_ref).resolve()
    if path == root or root not in path.parents:
        raise RuntimeToolError("unsafe_ledger_path")
    return path


def identifier_blocker(call: RuntimeToolCall) -> str | None:
    if safe_ledger_segment(call.run_id, "invalid-run-id") != call.run_id:
        return "invalid_run_id"
    if safe_ledger_segment(call.tool_name, "invalid-tool") != call.tool_name:
        return "invalid_tool_name"
    return None


def safe_ledger_segment(value: str, fallback: str) -> str:
    return value if SAFE_LEDGER_SEGMENT_RE.fullmatch(value) else fallback


def ledger_ref(call: RuntimeToolCall) -> str:
    run_id = safe_ledger_segment(call.run_id, "invalid-run-id")
    tool_name = safe_ledger_segment(call.tool_name, "invalid-tool")
    return f"tool_ledgers/{run_id}/{tool_name}.json"
