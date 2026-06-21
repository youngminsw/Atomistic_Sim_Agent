from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from sim_agent.schemas._parse import JsonMap

from .agent_registry import AgentSessionHandle, load_agent_registry


AgentMessageRole = Literal["user", "assistant", "system"]


def append_agent_message(
    session_dir: Path,
    agent_id: str,
    role: AgentMessageRole,
    content: str,
) -> AgentSessionHandle:
    handle = load_agent_registry(session_dir).handles[agent_id]
    sequence = _next_sequence(handle.messages_path)
    payload: JsonMap = {
        "at": time.time(),
        "sequence": sequence,
        "agent_id": handle.agent_id,
        "agent_session_id": handle.agent_session_id,
        "role": role,
        "content": content,
    }
    handle.messages_path.parent.mkdir(parents=True, exist_ok=True)
    with handle.messages_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    append_agent_event(session_dir, agent_id, "agent_message_appended", f"{role} message appended")
    from .compaction import auto_compact_agent_session

    auto_compact_agent_session(session_dir, agent_id)
    return handle


def append_agent_event(
    session_dir: Path,
    agent_id: str,
    event_type: str,
    summary: str,
) -> AgentSessionHandle:
    handle = load_agent_registry(session_dir).handles[agent_id]
    sequence = _next_sequence(handle.events_path)
    payload: JsonMap = {
        "at": time.time(),
        "sequence": sequence,
        "agent_id": handle.agent_id,
        "agent_session_id": handle.agent_session_id,
        "event_type": event_type,
        "summary": summary,
    }
    handle.events_path.parent.mkdir(parents=True, exist_ok=True)
    with handle.events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return handle


def _next_sequence(path: Path) -> int:
    if not path.is_file():
        return 1
    return len(path.read_text(encoding="utf-8").splitlines()) + 1
