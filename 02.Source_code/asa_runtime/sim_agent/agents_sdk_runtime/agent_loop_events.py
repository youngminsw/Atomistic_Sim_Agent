from __future__ import annotations

import time

from sim_agent.schemas._parse import JsonMap

from .types import RuntimeEvent, RuntimeEventType


class AgentLoopEventStream:
    def __init__(self, *, session_id: str, agent_id: str) -> None:
        self._session_id = session_id
        self._agent_id = agent_id
        self._sequence = 0
        self._events: list[RuntimeEvent] = []

    def append(
        self,
        event_type: RuntimeEventType,
        turn_id: str,
        payload: JsonMap,
        *,
        correlation_id: str = "",
        parent_id: str = "",
    ) -> RuntimeEvent:
        self._sequence += 1
        event = RuntimeEvent(
            sequence=self._sequence,
            at=time.time(),
            session_id=self._session_id,
            turn_id=turn_id,
            event_type=event_type,
            agent_id=self._agent_id,
            payload=payload,
            correlation_id=correlation_id,
            parent_id=parent_id,
        )
        self._events.append(event)
        return event

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)


def loop_turn_id(session_id: str) -> str:
    return f"{session_id}:loop"


def model_turn_id(session_id: str, step: int) -> str:
    return f"{session_id}:model:{step}"


def tool_call_id(turn_id: str, index: int, tool_name: str) -> str:
    return f"{turn_id}:tool:{index}:{tool_name}"
