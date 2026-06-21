from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .agent_registry import load_agent_registry
from .agent_session_io import append_agent_event
from .live_agent_turn import dispatch_live_agent_message
from .message_bus import SendAgentMessageRequest


HANDOFF_LEDGER_NAME = "handoffs.jsonl"
HANDOFF_ERROR_NAME = "handoff_errors.jsonl"
HANDOFF_TIMEOUT_S = 1800


@dataclass(frozen=True, slots=True)
class HandoffTaskRequest:
    from_agent: str
    target_agent: str
    task_id: str
    thread_id: str
    task: str


@dataclass(frozen=True, slots=True)
class HandoffTaskResult:
    status: str
    handoff_status: str
    task_id: str
    target_agent: str
    artifact_ref: str
    blocker: str | None = None

    def to_json(self) -> JsonMap:
        return {
            "status": self.handoff_status,
            "task_id": self.task_id,
            "target_agent": self.target_agent,
            "artifact_ref": self.artifact_ref,
            "blocker": self.blocker or "",
            "timeout_s": HANDOFF_TIMEOUT_S,
        }


def handoff_task(session_dir: Path, request: HandoffTaskRequest) -> HandoffTaskResult:
    blocker = _handoff_blocker(session_dir, request)
    if blocker is not None:
        return _blocked(session_dir, request, blocker)
    dispatch = dispatch_live_agent_message(
        session_dir,
        SendAgentMessageRequest(
            from_agent=request.from_agent,
            to_agent=request.target_agent,
            content=request.task,
            thread_id=request.thread_id,
            message_id=f"handoff-{request.task_id}",
        ),
    )
    if dispatch.blockers:
        return _blocked(session_dir, request, dispatch.blockers[0])
    append_agent_event(session_dir, request.target_agent, "handoff_task_executed", f"handoff {request.task_id} live turn {dispatch.turn_status}")
    payload: JsonMap = {
        "at": time.time(),
        "task_id": request.task_id,
        "thread_id": request.thread_id,
        "from": request.from_agent,
        "target_agent": request.target_agent,
        "agent_session_id": dispatch.agent_session_id,
        "status": "live_completed",
        "agent_loop_status": dispatch.turn_status,
        "selected_tools": list(dispatch.selected_tools),
        "timeout_s": HANDOFF_TIMEOUT_S,
        "message_id": dispatch.message_id,
        "bus_statuses": list(dispatch.bus_statuses),
    }
    _append_jsonl(session_dir / "message_bus" / HANDOFF_LEDGER_NAME, payload)
    return HandoffTaskResult("succeeded", "live_completed", request.task_id, request.target_agent, f"message_bus/{HANDOFF_LEDGER_NAME}")


def _handoff_blocker(session_dir: Path, request: HandoffTaskRequest) -> str | None:
    registry = load_agent_registry(session_dir)
    if request.from_agent not in registry.handles or request.target_agent not in registry.handles:
        return "unknown_agent"
    return None


def _blocked(session_dir: Path, request: HandoffTaskRequest, blocker: str) -> HandoffTaskResult:
    payload: JsonMap = {
        "at": time.time(),
        "task_id": request.task_id,
        "thread_id": request.thread_id,
        "from": request.from_agent,
        "target_agent": request.target_agent,
        "status": "blocked",
        "blocker": blocker,
        "timeout_s": HANDOFF_TIMEOUT_S,
    }
    _append_jsonl(session_dir / "message_bus" / HANDOFF_ERROR_NAME, payload)
    return HandoffTaskResult("blocked", "blocked", request.task_id, request.target_agent, f"message_bus/{HANDOFF_ERROR_NAME}", blocker)


def _append_jsonl(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
