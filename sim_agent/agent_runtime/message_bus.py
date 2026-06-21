from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from sim_agent.schemas._parse import JsonMap

from .agent_registry import load_agent_registry


MESSAGE_BUS_DIR_NAME = "message_bus"
MESSAGE_BUS_LEDGER_NAME = "messages.jsonl"
MESSAGE_BUS_ERROR_NAME = "message_bus_errors.jsonl"
MESSAGE_BUS_LOCK_NAME = "agent_message.lock"
MESSAGE_BUS_TIMEOUT_S = 1800
USER_MESSAGE_SENDER: Final = "user"
BusRecordType = Literal["send", "ack", "read", "reply"]


@dataclass(frozen=True, slots=True)
class AgentMessageBusResult:
    status: str
    bus_status: str
    message_id: str
    thread_id: str
    artifact_ref: str
    blocker: str | None = None

    def to_json(self) -> JsonMap:
        return {
            "status": self.bus_status,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "artifact_ref": self.artifact_ref,
            "blocker": self.blocker or "",
            "timeout_s": MESSAGE_BUS_TIMEOUT_S,
        }


@dataclass(frozen=True, slots=True)
class SendAgentMessageRequest:
    from_agent: str
    to_agent: str
    content: str
    thread_id: str
    message_id: str
    blocked_targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplyAgentMessageRequest:
    message_id: str
    by_agent: str
    content: str


@dataclass(frozen=True, slots=True)
class MessageStateRequest:
    record_type: BusRecordType
    message_id: str
    by_agent: str
    status: str
    content: str = ""


@dataclass(frozen=True, slots=True)
class BlockedMessageRequest:
    blocker: str
    message_id: str
    thread_id: str


@dataclass(frozen=True, slots=True)
class BusRecordDraft:
    record_type: BusRecordType
    message_id: str
    thread_id: str
    from_agent: str
    to_agent: str
    status: str


@dataclass(frozen=True, slots=True)
class FindMessageResult:
    status: str
    payload: JsonMap | None = None
    blocker: str | None = None


def send_agent_message(
    session_dir: Path,
    request: SendAgentMessageRequest,
) -> AgentMessageBusResult:
    blocker = _send_blocker(session_dir, request)
    if blocker is not None:
        return _blocked(session_dir, BlockedMessageRequest(blocker, request.message_id, request.thread_id))
    payload = _base_record(
        BusRecordDraft("send", request.message_id, request.thread_id, request.from_agent, request.to_agent, "sent")
    )
    payload.update({"content": request.content, "acknowledged_at": 0.0, "read_at": 0.0, "replied_at": 0.0})
    _append_bus_record(session_dir, payload)
    return AgentMessageBusResult("succeeded", "sent", request.message_id, request.thread_id, _bus_ref())


def ack_agent_message(session_dir: Path, *, message_id: str, by_agent: str) -> AgentMessageBusResult:
    return _append_state_record(session_dir, MessageStateRequest("ack", message_id, by_agent, "acknowledged"))


def read_agent_message(session_dir: Path, *, message_id: str, by_agent: str) -> AgentMessageBusResult:
    return _append_state_record(session_dir, MessageStateRequest("read", message_id, by_agent, "read"))


def reply_agent_message(
    session_dir: Path,
    request: ReplyAgentMessageRequest,
) -> AgentMessageBusResult:
    return _append_state_record(
        session_dir,
        MessageStateRequest("reply", request.message_id, request.by_agent, "replied", request.content),
    )


def _append_state_record(
    session_dir: Path,
    request: MessageStateRequest,
) -> AgentMessageBusResult:
    found = _find_message(session_dir, request.message_id)
    if found.blocker is not None:
        return _blocked(session_dir, BlockedMessageRequest(found.blocker, request.message_id, ""))
    if found.payload is None:
        return _blocked(session_dir, BlockedMessageRequest("message_not_found", request.message_id, ""))
    blocker = _known_agent_blocker(session_dir, request.by_agent)
    if blocker is not None:
        return _blocked(session_dir, BlockedMessageRequest(blocker, request.message_id, _thread_id(found.payload)))
    if request.by_agent != _to_agent(found.payload):
        return _blocked(session_dir, BlockedMessageRequest("wrong_recipient", request.message_id, _thread_id(found.payload)))
    payload = _base_record(
        BusRecordDraft(request.record_type, request.message_id, _thread_id(found.payload), request.by_agent, _to_agent(found.payload), request.status)
    )
    if request.content:
        payload["content"] = request.content
    match request.record_type:
        case "ack":
            payload["acknowledged_at"] = payload["at"]
        case "read":
            payload["read_at"] = payload["at"]
        case "reply":
            payload["replied_at"] = payload["at"]
        case "send":
            pass
    _append_bus_record(session_dir, payload)
    return AgentMessageBusResult("succeeded", request.status, request.message_id, _thread_id(found.payload), _bus_ref())


def _send_blocker(
    session_dir: Path,
    request: SendAgentMessageRequest,
) -> str | None:
    bus_dir = session_dir / MESSAGE_BUS_DIR_NAME
    if not bus_dir.is_dir():
        return "message_bus_missing"
    if (bus_dir / MESSAGE_BUS_LOCK_NAME).exists():
        return "stale_lock"
    if request.from_agent != USER_MESSAGE_SENDER and _known_agent_blocker(session_dir, request.from_agent) is not None:
        return "unknown_agent"
    if _known_agent_blocker(session_dir, request.to_agent) is not None:
        return "unknown_agent"
    if request.to_agent in request.blocked_targets:
        return "blocked_target"
    found = _find_message(session_dir, request.message_id)
    if found.blocker is not None:
        return found.blocker
    if found.payload is not None:
        return "duplicate_message_id"
    return None


def _known_agent_blocker(session_dir: Path, agent_id: str) -> str | None:
    registry = load_agent_registry(session_dir)
    if agent_id not in registry.handles:
        return "unknown_agent"
    return None


def _append_bus_record(session_dir: Path, payload: JsonMap) -> None:
    path = _bus_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _blocked(session_dir: Path, request: BlockedMessageRequest) -> AgentMessageBusResult:
    payload: JsonMap = {
        "at": time.time(),
        "status": "blocked",
        "blocker": request.blocker,
        "message_id": request.message_id,
        "thread_id": request.thread_id,
        "timeout_s": MESSAGE_BUS_TIMEOUT_S,
    }
    with (session_dir / MESSAGE_BUS_ERROR_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return AgentMessageBusResult("blocked", "blocked", request.message_id, request.thread_id, MESSAGE_BUS_ERROR_NAME, request.blocker)


def _base_record(draft: BusRecordDraft) -> dict[str, object]:
    return {
        "at": time.time(),
        "record_type": draft.record_type,
        "message_id": draft.message_id,
        "thread_id": draft.thread_id,
        "from": draft.from_agent,
        "to": draft.to_agent,
        "status": draft.status,
        "timeout_s": MESSAGE_BUS_TIMEOUT_S,
    }


def _find_message(session_dir: Path, message_id: str) -> FindMessageResult:
    path = _bus_path(session_dir)
    if not path.is_file():
        return FindMessageResult("missing")
    candidate: JsonMap | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return FindMessageResult("blocked", blocker="corrupt_message_bus")
        if isinstance(value, dict) and value.get("record_type") == "send" and value.get("message_id") == message_id:
            candidate = value
    if candidate is not None:
        return FindMessageResult("found", candidate)
    return FindMessageResult("missing")


def _thread_id(payload: JsonMap) -> str:
    value = payload.get("thread_id")
    return value if isinstance(value, str) and value else ""


def _to_agent(payload: JsonMap) -> str:
    value = payload.get("to")
    return value if isinstance(value, str) and value else ""


def _bus_path(session_dir: Path) -> Path:
    return session_dir / MESSAGE_BUS_DIR_NAME / MESSAGE_BUS_LEDGER_NAME


def _bus_ref() -> str:
    return f"{MESSAGE_BUS_DIR_NAME}/{MESSAGE_BUS_LEDGER_NAME}"
