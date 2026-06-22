from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_runtime.live_agent_turn import dispatch_live_agent_message
from sim_agent.agent_runtime.message_bus import SendAgentMessageRequest


@dataclass(frozen=True, slots=True)
class DirectAgentChatRequest:
    target: str
    message: str
    session_id: str
    session_dir: Path


@dataclass(frozen=True, slots=True)
class DirectAgentChatResult:
    target: str
    agent_session_id: str
    agent_session_path: Path
    turn_status: str
    model_id: str
    selected_tools: tuple[str, ...]

    @property
    def assistant_content(self) -> str:
        return f"{self.target} completed a live agent loop in persistent session {self.agent_session_id}."


def run_direct_agent_chat(request: DirectAgentChatRequest) -> DirectAgentChatResult:
    dispatch = dispatch_live_agent_message(
        request.session_dir,
        SendAgentMessageRequest(
            from_agent="user",
            to_agent=request.target,
            content=request.message,
            thread_id=f"direct-{request.session_id}-{request.target}",
            message_id=f"direct-{request.target}-{time.time_ns()}",
        ),
    )
    return DirectAgentChatResult(
        target=request.target,
        agent_session_id=dispatch.agent_session_id,
        agent_session_path=request.session_dir / "agent_sessions" / request.target,
        turn_status=dispatch.turn_status,
        model_id=dispatch.model_id,
        selected_tools=dispatch.selected_tools,
    )
