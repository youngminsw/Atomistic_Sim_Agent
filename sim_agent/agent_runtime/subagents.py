from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.errors import SchemaValidationError

from .agent_registry import load_agent_registry
from .agent_specs import SubagentPresetSpec, resolve_subagent_preset
from .agent_session_io import append_agent_event
from .subagent_loop import SubagentLoopRun, run_subagent_agent_loop


MAX_ACTIVE_SUBAGENTS_PER_CALLER: Final = 4
SUBAGENT_RUN_LEDGER_NAME: Final = "subagent_run.json"
SUBAGENT_RUNNING_LOCK_NAME: Final = "subagent_running.lock"
SAFE_SUBAGENT_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True, slots=True)
class SubagentTaskRequest:
    caller_agent: str
    preset: str
    task_id: str
    task: str
    depth: int = 1


@dataclass(frozen=True, slots=True)
class SubagentInspectRequest:
    caller_agent: str
    preset: str
    subagent_id: str


@dataclass(frozen=True, slots=True)
class SubagentTaskResult:
    status: str
    subagent_status: str
    caller_agent: str
    preset: str
    subagent_id: str
    session_dir: Path
    artifact_ref: str
    depth: int
    tool_names: tuple[str, ...]
    model_id: str
    selected_tools: tuple[str, ...]
    blockers: tuple[str, ...]
    blocker: str | None = None

    def to_json(self) -> JsonMap:
        return {
            "status": self.subagent_status,
            "agent_loop_status": self.subagent_status,
            "caller_agent": self.caller_agent,
            "preset": self.preset,
            "subagent_id": self.subagent_id,
            "depth": self.depth,
            "session_dir": str(self.session_dir),
            "tool_names": list(self.tool_names),
            "model_id": self.model_id,
            "selected_tools": list(self.selected_tools),
            "blockers": list(self.blockers),
            "artifact_ref": self.artifact_ref,
            "blocker": self.blocker or "",
        }


@dataclass(frozen=True, slots=True)
class SubagentInspectResult:
    status: str
    output: JsonMap
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class SubagentLocator:
    caller_agent: str
    preset: str
    subagent_id: str


def run_bounded_subagent(session_dir: Path, request: SubagentTaskRequest) -> SubagentTaskResult:
    blocker = _task_blocker(session_dir, request)
    if blocker is not None:
        return _blocked_task(session_dir, request, blocker)
    preset = resolve_subagent_preset(request.preset)
    handle = load_agent_registry(session_dir).handles[request.caller_agent]
    locator = SubagentLocator(request.caller_agent, preset.name, request.task_id)
    subagent_dir = _subagent_dir(session_dir, locator)
    subagent_dir.mkdir(parents=True, exist_ok=False)
    running_lock = subagent_dir / SUBAGENT_RUNNING_LOCK_NAME
    running_lock.write_text(json.dumps(_running_lock_payload(request), sort_keys=True) + "\n", encoding="utf-8")
    try:
        loop_run = run_subagent_agent_loop(handle, preset, request.task_id, request.task, request.depth, subagent_dir)
    finally:
        running_lock.unlink(missing_ok=True)
    payload = _run_payload(request, preset, subagent_dir, loop_run)
    _write_json(subagent_dir / SUBAGENT_RUN_LEDGER_NAME, payload)
    _append_jsonl(subagent_dir / "messages.jsonl", _message_payload("user", request.task))
    _append_jsonl(subagent_dir / "messages.jsonl", _message_payload("assistant", f"{preset.name} bounded run {loop_run.status}"))
    append_agent_event(session_dir, request.caller_agent, "subagent_task_executed", f"{preset.name} subagent {request.task_id} {loop_run.status}")
    blocker = loop_run.blockers[0] if loop_run.blockers else None
    status = "succeeded" if loop_run.status == "succeeded" else "blocked"
    return SubagentTaskResult(
        status=status,
        subagent_status=loop_run.status,
        caller_agent=request.caller_agent,
        preset=preset.name,
        subagent_id=request.task_id,
        session_dir=subagent_dir,
        artifact_ref=_artifact_ref(locator),
        depth=request.depth,
        tool_names=preset.tool_names,
        model_id=loop_run.model_id,
        selected_tools=loop_run.selected_tools,
        blockers=loop_run.blockers,
        blocker=blocker,
    )


def inspect_bounded_subagent(session_dir: Path, request: SubagentInspectRequest) -> SubagentInspectResult:
    try:
        preset = resolve_subagent_preset(request.preset)
    except SchemaValidationError:
        return SubagentInspectResult("blocked", _inspect_error(request, "unknown_preset"), "unknown_preset")
    subagent_dir = _subagent_dir(session_dir, SubagentLocator(request.caller_agent, preset.name, request.subagent_id))
    ledger_path = subagent_dir / SUBAGENT_RUN_LEDGER_NAME
    if not ledger_path.is_file():
        return SubagentInspectResult("blocked", _inspect_error(request, "unknown_subagent"), "unknown_subagent")
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return SubagentInspectResult("blocked", _inspect_error(request, "corrupt_subagent_ledger"), "corrupt_subagent_ledger")
    return SubagentInspectResult("succeeded", payload)


def _task_blocker(session_dir: Path, request: SubagentTaskRequest) -> str | None:
    if not SAFE_SUBAGENT_ID_RE.fullmatch(request.task_id):
        return "invalid_task_id"
    if request.depth > 1:
        return "subagent_depth_exceeded"
    if request.caller_agent == request.preset:
        return "subagent_recursion_blocked"
    try:
        preset = resolve_subagent_preset(request.preset)
    except SchemaValidationError:
        return "unknown_preset"
    registry = load_agent_registry(session_dir)
    if request.caller_agent not in registry.handles:
        return "unknown_agent"
    if _task_id_exists(session_dir, request.caller_agent, request.task_id):
        return "duplicate_task_id"
    if _active_child_count(session_dir, request.caller_agent) >= MAX_ACTIVE_SUBAGENTS_PER_CALLER:
        return "too_many_active_subagents"
    if request.depth > preset.max_depth:
        return "subagent_depth_exceeded"
    return None


def _blocked_task(session_dir: Path, request: SubagentTaskRequest, blocker: str) -> SubagentTaskResult:
    preset_name = request.preset if request.preset else "unknown"
    locator = SubagentLocator(request.caller_agent, preset_name, request.task_id)
    subagent_dir = _subagent_dir(session_dir, locator)
    return SubagentTaskResult(
        status="blocked",
        subagent_status="blocked",
        caller_agent=request.caller_agent,
        preset=preset_name,
        subagent_id=request.task_id,
        session_dir=subagent_dir,
        artifact_ref=_artifact_ref(locator),
        depth=request.depth,
        tool_names=(),
        model_id="",
        selected_tools=(),
        blockers=(blocker,),
        blocker=blocker,
    )


def _run_payload(
    request: SubagentTaskRequest,
    preset: SubagentPresetSpec,
    subagent_dir: Path,
    loop_run: SubagentLoopRun,
) -> JsonMap:
    return {
        "schema_version": "asa_subagent_run_v1",
        "at": time.time(),
        "caller_agent": request.caller_agent,
        "preset": preset.name,
        "subagent_id": request.task_id,
        "task": request.task,
        "depth": request.depth,
        "status": loop_run.status,
        "agent_loop": {
            "status": loop_run.status,
            "model_id": loop_run.model_id,
            "selected_tools": list(loop_run.selected_tools),
            "blockers": list(loop_run.blockers),
            "trace": list(loop_run.trace),
            "tool_result_refs": list(loop_run.tool_result_refs),
        },
        "session_dir": str(subagent_dir),
        "tool_names": list(preset.tool_names),
        "preset_spec": asdict(preset),
    }


def _message_payload(role: str, content: str) -> JsonMap:
    return {"at": time.time(), "role": role, "content": content}


def _running_lock_payload(request: SubagentTaskRequest) -> JsonMap:
    return {
        "schema_version": "asa_subagent_running_lock_v1",
        "at": time.time(),
        "caller_agent": request.caller_agent,
        "preset": request.preset,
        "subagent_id": request.task_id,
        "depth": request.depth,
    }


def _inspect_error(request: SubagentInspectRequest, blocker: str) -> JsonMap:
    return {
        "caller_agent": request.caller_agent,
        "preset": request.preset,
        "subagent_id": request.subagent_id,
        "blocker": blocker,
    }


def _task_id_exists(session_dir: Path, caller_agent: str, task_id: str) -> bool:
    subagents_dir = session_dir / "agent_sessions" / caller_agent / "subagents"
    return any(path.name == task_id for path in subagents_dir.glob("*/*")) if subagents_dir.is_dir() else False


def _active_child_count(session_dir: Path, caller_agent: str) -> int:
    subagents_dir = session_dir / "agent_sessions" / caller_agent / "subagents"
    if not subagents_dir.is_dir():
        return 0
    return sum(1 for path in subagents_dir.glob("*/*") if (path / SUBAGENT_RUNNING_LOCK_NAME).is_file())


def _subagent_dir(session_dir: Path, locator: SubagentLocator) -> Path:
    return session_dir / "agent_sessions" / locator.caller_agent / "subagents" / locator.preset / locator.subagent_id


def _artifact_ref(locator: SubagentLocator) -> str:
    return f"agent_sessions/{locator.caller_agent}/subagents/{locator.preset}/{locator.subagent_id}/{SUBAGENT_RUN_LEDGER_NAME}"


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: JsonMap) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
