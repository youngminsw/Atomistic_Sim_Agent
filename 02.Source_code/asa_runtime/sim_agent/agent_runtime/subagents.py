from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.errors import SchemaValidationError

from .agent_registry import AgentSessionHandle, load_agent_registry
from .agent_specs import SubagentPresetSpec, resolve_subagent_preset
from .agent_session_io import append_agent_event
from .subagent_loop import SubagentLoopRun, run_subagent_agent_loop


MAX_ACTIVE_SUBAGENTS_PER_CALLER: Final = 4
SUBAGENT_RUN_LEDGER_NAME: Final = "subagent_run.json"
SUBAGENT_RUNNING_LOCK_NAME: Final = "subagent_running.lock"
SUBAGENT_CONTROL_LEDGER_NAME: Final = "subagent_controls.jsonl"
SUBAGENT_CONTROL_STATE_NAME: Final = "subagent_control_state.json"
SAFE_SUBAGENT_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
SUBAGENT_CONTROL_ACTIONS: Final = frozenset(
    {"list", "progress", "await", "cancel", "pause", "resume", "steer", "restart"}
)


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
class SubagentControlRequest:
    action: str
    caller_agent: str
    preset: str = ""
    subagent_id: str = ""
    content: str = ""


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
class SubagentControlResult:
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
    payload = _run_payload(request, preset, handle, subagent_dir, loop_run)
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


def control_bounded_subagent(session_dir: Path, request: SubagentControlRequest) -> SubagentControlResult:
    if request.action not in SUBAGENT_CONTROL_ACTIONS:
        return SubagentControlResult("blocked", _control_error(request, "invalid_action"), "invalid_action")
    registry = load_agent_registry(session_dir)
    if request.caller_agent not in registry.handles:
        return SubagentControlResult("blocked", _control_error(request, "unknown_agent"), "unknown_agent")
    if request.action == "list":
        return SubagentControlResult("succeeded", _list_output(session_dir, request.caller_agent))
    located = _control_locator(session_dir, request)
    if isinstance(located, SubagentControlResult):
        return located
    subagent_dir, ledger = located
    if request.action in {"progress", "await"}:
        output = _control_snapshot(request, subagent_dir, ledger)
        if _subagent_lost_process(subagent_dir):
            return SubagentControlResult("blocked", output, "subagent_lost_process")
        if request.action == "await" and (subagent_dir / SUBAGENT_RUNNING_LOCK_NAME).is_file():
            return SubagentControlResult("blocked", output, "subagent_still_running")
        return SubagentControlResult("succeeded", output)
    if request.action == "restart":
        return _restart_lost_subagent(session_dir, request, subagent_dir)
    if request.action in {"cancel", "pause", "resume", "steer"}:
        if not (subagent_dir / SUBAGENT_RUNNING_LOCK_NAME).is_file():
            return SubagentControlResult("blocked", _control_snapshot(request, subagent_dir, ledger, "subagent_already_terminal"), "subagent_already_terminal")
        if _subagent_lost_process(subagent_dir):
            return SubagentControlResult("blocked", _control_snapshot(request, subagent_dir, ledger, "subagent_lost_process"), "subagent_lost_process")
        _append_control_event(subagent_dir, request)
        if request.action in {"pause", "resume", "cancel"}:
            _write_control_state(subagent_dir, request)
        return SubagentControlResult("succeeded", _control_snapshot(request, subagent_dir, ledger))
    return SubagentControlResult("blocked", _control_error(request, "invalid_action"), "invalid_action")


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
    handle: AgentSessionHandle,
    subagent_dir: Path,
    loop_run: SubagentLoopRun,
) -> JsonMap:
    return {
        "schema_version": "asa_subagent_run_v1",
        "at": time.time(),
        "caller_agent": request.caller_agent,
        "preset": preset.name,
        "subagent_id": request.task_id,
        "owner": {
            "caller_agent": request.caller_agent,
            "caller_agent_session_id": handle.agent_session_id,
            "caller_agent_session_dir": str(handle.session_dir),
        },
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
        "role_prompt_layer": {
            "kind": "subagent_role",
            "caller_agent": request.caller_agent,
            "preset": preset.name,
            "role_prompt": preset.role_prompt,
            "scope_notes": preset.scope_notes,
        },
        "lifecycle": {
            "state": "completed" if loop_run.status == "succeeded" else "blocked",
            "running": False,
            "controllable": False,
        },
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
        "task": request.task,
        "owner_pid": os.getpid(),
    }


def _inspect_error(request: SubagentInspectRequest, blocker: str) -> JsonMap:
    return {
        "caller_agent": request.caller_agent,
        "preset": request.preset,
        "subagent_id": request.subagent_id,
        "blocker": blocker,
    }


def _control_error(request: SubagentControlRequest, blocker: str) -> JsonMap:
    return {
        "action": request.action,
        "caller_agent": request.caller_agent,
        "preset": request.preset,
        "subagent_id": request.subagent_id,
        "blocker": blocker,
    }


def _list_output(session_dir: Path, caller_agent: str) -> JsonMap:
    subagents_dir = session_dir / "agent_sessions" / caller_agent / "subagents"
    jobs: list[JsonMap] = []
    if subagents_dir.is_dir():
        for ledger_path in sorted(subagents_dir.glob(f"*/*/{SUBAGENT_RUN_LEDGER_NAME}")):
            payload = _read_json(ledger_path)
            if isinstance(payload, dict):
                jobs.append(_job_summary(ledger_path.parent, payload))
        for lock_path in sorted(subagents_dir.glob(f"*/*/{SUBAGENT_RUNNING_LOCK_NAME}")):
            if not (lock_path.parent / SUBAGENT_RUN_LEDGER_NAME).is_file():
                lock = _read_json(lock_path)
                if isinstance(lock, dict):
                    jobs.append(_running_job_summary(lock_path.parent, lock))
    return {"caller_agent": caller_agent, "subagents": jobs}


def _control_locator(
    session_dir: Path,
    request: SubagentControlRequest,
) -> tuple[Path, JsonMap] | SubagentControlResult:
    if not request.preset:
        return SubagentControlResult("blocked", _control_error(request, "preset_required"), "preset_required")
    if not request.subagent_id:
        return SubagentControlResult("blocked", _control_error(request, "subagent_id_required"), "subagent_id_required")
    try:
        preset = resolve_subagent_preset(request.preset)
    except SchemaValidationError:
        return SubagentControlResult("blocked", _control_error(request, "unknown_preset"), "unknown_preset")
    subagent_dir = _subagent_dir(session_dir, SubagentLocator(request.caller_agent, preset.name, request.subagent_id))
    ledger_path = subagent_dir / SUBAGENT_RUN_LEDGER_NAME
    if ledger_path.is_file():
        payload = _read_json(ledger_path)
        if isinstance(payload, dict):
            return subagent_dir, payload
        return SubagentControlResult("blocked", _control_error(request, "corrupt_subagent_ledger"), "corrupt_subagent_ledger")
    lock_path = subagent_dir / SUBAGENT_RUNNING_LOCK_NAME
    if lock_path.is_file():
        lock = _read_json(lock_path)
        if isinstance(lock, dict):
            return subagent_dir, {
                "schema_version": "asa_subagent_run_v1",
                "caller_agent": request.caller_agent,
                "preset": preset.name,
                "subagent_id": request.subagent_id,
                "status": "running",
                "task": "",
                "session_dir": str(subagent_dir),
            }
    return SubagentControlResult("blocked", _control_error(request, "unknown_subagent"), "unknown_subagent")


def _control_snapshot(
    request: SubagentControlRequest,
    subagent_dir: Path,
    ledger: JsonMap,
    blocker: str | None = None,
) -> JsonMap:
    running = (subagent_dir / SUBAGENT_RUNNING_LOCK_NAME).is_file()
    lost_process = _subagent_lost_process(subagent_dir)
    state = _control_state(subagent_dir, running, lost_process)
    output = {
        "action": request.action,
        "caller_agent": request.caller_agent,
        "preset": request.preset,
        "subagent_id": request.subagent_id,
        "status": ledger.get("status", "running"),
        "state": state,
        "running": running,
        "controllable": running,
        "lost_process": lost_process,
        "session_dir": str(subagent_dir),
        "artifact_ref": _artifact_ref(SubagentLocator(request.caller_agent, request.preset, request.subagent_id)),
        "control_events": _control_events(subagent_dir),
    }
    if blocker is not None:
        output["blocker"] = blocker
    return output


def _job_summary(subagent_dir: Path, payload: JsonMap) -> JsonMap:
    caller_agent = str(payload.get("caller_agent", ""))
    preset = str(payload.get("preset", ""))
    subagent_id = str(payload.get("subagent_id", ""))
    return {
        "caller_agent": caller_agent,
        "preset": preset,
        "subagent_id": subagent_id,
        "status": payload.get("status", ""),
        "state": _control_state(subagent_dir, False),
        "running": False,
        "controllable": False,
        "session_dir": str(subagent_dir),
        "artifact_ref": _artifact_ref(SubagentLocator(caller_agent, preset, subagent_id)),
    }


def _running_job_summary(subagent_dir: Path, lock: JsonMap) -> JsonMap:
    caller_agent = str(lock.get("caller_agent", ""))
    preset = str(lock.get("preset", ""))
    subagent_id = str(lock.get("subagent_id", ""))
    lost_process = _lock_owner_lost(lock)
    return {
        "caller_agent": caller_agent,
        "preset": preset,
        "subagent_id": subagent_id,
        "status": "running",
        "state": _control_state(subagent_dir, True, lost_process),
        "running": True,
        "controllable": True,
        "lost_process": lost_process,
        "session_dir": str(subagent_dir),
        "artifact_ref": _artifact_ref(SubagentLocator(caller_agent, preset, subagent_id)),
    }


def _append_control_event(subagent_dir: Path, request: SubagentControlRequest) -> None:
    _append_jsonl(
        subagent_dir / SUBAGENT_CONTROL_LEDGER_NAME,
        {
            "schema_version": "asa_subagent_control_event_v1",
            "at": time.time(),
            "action": request.action,
            "caller_agent": request.caller_agent,
            "preset": request.preset,
            "subagent_id": request.subagent_id,
            "content": request.content,
        },
    )


def _write_control_state(subagent_dir: Path, request: SubagentControlRequest) -> None:
    state_by_action = {"pause": "paused", "resume": "running", "cancel": "cancel_requested"}
    state = state_by_action[request.action]
    _write_json(
        subagent_dir / SUBAGENT_CONTROL_STATE_NAME,
        {
            "schema_version": "asa_subagent_control_state_v1",
            "at": time.time(),
            "state": state,
            "last_action": request.action,
        },
    )


def _restart_lost_subagent(session_dir: Path, request: SubagentControlRequest, subagent_dir: Path) -> SubagentControlResult:
    lock_path = subagent_dir / SUBAGENT_RUNNING_LOCK_NAME
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        return SubagentControlResult("blocked", _control_error(request, "unknown_subagent"), "unknown_subagent")
    if not _lock_owner_lost(lock):
        return SubagentControlResult("blocked", _control_snapshot(request, subagent_dir, lock, "subagent_still_running"), "subagent_still_running")
    task = str(lock.get("task") or request.content or "")
    if not task:
        return SubagentControlResult("blocked", _control_snapshot(request, subagent_dir, lock, "restart_task_missing"), "restart_task_missing")
    depth = lock.get("depth", 1)
    if not isinstance(depth, int) or isinstance(depth, bool):
        depth = 1
    archive_dir = subagent_dir.with_name(f"{subagent_dir.name}.lost-{time.time_ns()}")
    subagent_dir.rename(archive_dir)
    restarted = run_bounded_subagent(
        session_dir,
        SubagentTaskRequest(
            caller_agent=request.caller_agent,
            preset=request.preset,
            task_id=request.subagent_id,
            task=task,
            depth=depth,
        ),
    )
    output = restarted.to_json()
    output["action"] = "restart"
    output["restarted_from"] = str(archive_dir)
    output["previous_blocker"] = "subagent_lost_process"
    _append_jsonl(
        subagent_dir / SUBAGENT_CONTROL_LEDGER_NAME,
        {
            "schema_version": "asa_subagent_control_event_v1",
            "at": time.time(),
            "action": "restart",
            "caller_agent": request.caller_agent,
            "preset": request.preset,
            "subagent_id": request.subagent_id,
            "content": request.content,
            "restarted_from": str(archive_dir),
        },
    )
    return SubagentControlResult(restarted.status, output, restarted.blocker)


def _control_state(subagent_dir: Path, running: bool, lost_process: bool = False) -> str:
    if lost_process:
        return "lost_process"
    payload = _read_json(subagent_dir / SUBAGENT_CONTROL_STATE_NAME)
    if isinstance(payload, dict):
        state = payload.get("state")
        if isinstance(state, str) and state:
            return state
    return "running" if running else "completed"


def _subagent_lost_process(subagent_dir: Path) -> bool:
    lock = _read_json(subagent_dir / SUBAGENT_RUNNING_LOCK_NAME)
    return _lock_owner_lost(lock) if isinstance(lock, dict) else False


def _lock_owner_lost(lock: JsonMap) -> bool:
    owner_pid = lock.get("owner_pid")
    if not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 0:
        return False
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _control_events(subagent_dir: Path) -> list[JsonMap]:
    path = subagent_dir / SUBAGENT_CONTROL_LEDGER_NAME
    if not path.is_file():
        return []
    events: list[JsonMap] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _read_json(path: Path) -> JsonMap | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
