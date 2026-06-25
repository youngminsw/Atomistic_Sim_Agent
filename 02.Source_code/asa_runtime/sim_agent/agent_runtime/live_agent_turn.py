from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .agent_registry import AgentSessionHandle, load_agent_registry
from .agent_session_io import append_agent_event, append_agent_message
from .compaction import COMPACT_SUMMARY_NAME
from .compaction_store import read_json, read_jsonl
from .live_agent_context import (
    live_turn_handle_with_model_override,
    live_turn_ledger_facts,
    live_turn_project_guidance,
    live_turn_skill_names,
    live_turn_workflow_policy,
    live_turn_workflow_state,
)
from .message_bus import (
    ReplyAgentMessageRequest,
    SendAgentMessageRequest,
    ack_agent_message,
    read_agent_message,
    reply_agent_message,
    send_agent_message,
)

if TYPE_CHECKING:
    from sim_agent.agents_sdk_runtime import AsaAgentSession, StaticToolChoiceModel, ToolChoiceModel


@dataclass(frozen=True, slots=True)
class LiveAgentTurnResult:
    agent_id: str
    agent_session_id: str
    status: str
    model_id: str
    selected_tools: tuple[str, ...]
    blockers: tuple[str, ...]
    runtime_events: tuple[JsonMap, ...] = ()

    def to_json(self) -> JsonMap:
        return {
            "agent_id": self.agent_id,
            "agent_session_id": self.agent_session_id,
            "status": self.status,
            "model_id": self.model_id,
            "selected_tools": list(self.selected_tools),
            "blockers": list(self.blockers),
            "runtime_events": list(self.runtime_events),
        }


@dataclass(frozen=True, slots=True)
class LiveAgentDispatchResult:
    agent_id: str
    agent_session_id: str
    message_id: str
    thread_id: str
    status: str
    turn_status: str
    model_id: str
    selected_tools: tuple[str, ...]
    bus_statuses: tuple[str, ...]
    blockers: tuple[str, ...]
    runtime_events: tuple[JsonMap, ...] = ()

    def to_json(self) -> JsonMap:
        return {
            "agent_id": self.agent_id,
            "agent_session_id": self.agent_session_id,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "turn_status": self.turn_status,
            "model_id": self.model_id,
            "selected_tools": list(self.selected_tools),
            "bus_statuses": list(self.bus_statuses),
            "blockers": list(self.blockers),
            "runtime_events": list(self.runtime_events),
        }


def dispatch_live_agent_message(
    session_dir: Path,
    request: SendAgentMessageRequest,
) -> LiveAgentDispatchResult:
    sent = send_agent_message(session_dir, request)
    if sent.blocker is not None:
        return _blocked_dispatch(request, sent.blocker, (sent.bus_status,))
    acked = ack_agent_message(session_dir, message_id=request.message_id, by_agent=request.to_agent)
    if acked.blocker is not None:
        return _blocked_dispatch(request, acked.blocker, (sent.bus_status, acked.bus_status))
    read = read_agent_message(session_dir, message_id=request.message_id, by_agent=request.to_agent)
    if read.blocker is not None:
        return _blocked_dispatch(request, read.blocker, (sent.bus_status, acked.bus_status, read.bus_status))
    handle = append_agent_message(session_dir, request.to_agent, "user", request.content)
    turn = run_live_agent_turn(session_dir, request.to_agent, request.content)
    replied = reply_agent_message(
        session_dir,
        ReplyAgentMessageRequest(
            message_id=request.message_id,
            by_agent=request.to_agent,
            content=f"{request.to_agent} live turn {turn.status}",
        ),
    )
    statuses = (sent.bus_status, acked.bus_status, read.bus_status, replied.bus_status)
    blockers = turn.blockers if replied.blocker is None else (*turn.blockers, replied.blocker)
    status = "succeeded" if not blockers and turn.status == "succeeded" else "blocked"
    return LiveAgentDispatchResult(
        agent_id=request.to_agent,
        agent_session_id=handle.agent_session_id,
        message_id=request.message_id,
        thread_id=request.thread_id,
        status=status,
        turn_status=turn.status,
        model_id=turn.model_id,
        selected_tools=turn.selected_tools,
        bus_statuses=statuses,
        blockers=blockers,
        runtime_events=turn.runtime_events,
    )


def run_live_agent_turn(session_dir: Path, agent_id: str, user_goal: str) -> LiveAgentTurnResult:
    from sim_agent.agents_sdk_runtime import AgentLoop, runtime_event_to_json

    handle = _ensure_turn_user_message(session_dir, agent_id, user_goal)
    runtime_handle = live_turn_handle_with_model_override(handle)
    session = _agent_loop_session(runtime_handle, user_goal)
    loop_result = AgentLoop(session, _live_turn_model(runtime_handle, user_goal, session)).run()
    for event in loop_result.trace:
        append_agent_event(session_dir, agent_id, event.event_type, event.summary)
    append_agent_message(session_dir, agent_id, "assistant", _assistant_message(loop_result.status, loop_result.blockers))
    return LiveAgentTurnResult(
        agent_id=agent_id,
        agent_session_id=handle.agent_session_id,
        status=loop_result.status,
        model_id=loop_result.model_id,
        selected_tools=tuple(call.tool_name for call in loop_result.selected_tools),
        blockers=loop_result.blockers,
        runtime_events=tuple(runtime_event_to_json(event) for event in loop_result.runtime_events),
    )


def _agent_loop_session(handle: AgentSessionHandle, user_goal: str) -> AsaAgentSession:
    from sim_agent.agent_harness.tools import tool_registry_for_agent
    from sim_agent.agents_sdk_runtime import AsaAgentSession

    return AsaAgentSession(
        run_id=_safe_run_id(handle.agent_id),
        session_id=handle.agent_session_id,
        agent_id=handle.agent_id,
        user_goal=user_goal,
        endpoint=_endpoint(handle),
        output_dir=handle.session_dir,
        registry=tool_registry_for_agent(handle.agent_id),
        role_prompt=handle.role_prompt,
        role_prompt_kind="domain_role",
        workflow_policy=live_turn_workflow_policy(),
        project_guidance=live_turn_project_guidance(handle),
        compact_summary=_compact_summary(handle),
        workflow_state=live_turn_workflow_state(handle),
        skills=live_turn_skill_names(handle.agent_id),
        ledger_facts=live_turn_ledger_facts(handle),
        messages=_chat_messages(handle),
    )


def _live_turn_model(handle: AgentSessionHandle, user_goal: str, session: AsaAgentSession) -> ToolChoiceModel:
    if _uses_static_fallback(handle):
        return _default_live_turn_model(handle, user_goal)

    from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel, provider_tool_choice_model_id

    return ProviderToolChoiceModel(model_id=provider_tool_choice_model_id(session))


def _default_live_turn_model(handle: AgentSessionHandle, user_goal: str) -> StaticToolChoiceModel:
    from sim_agent.agents_sdk_runtime import ModelSelectedToolCall, StaticToolChoiceModel

    content = f"{handle.agent_id} live agent loop accepted: {user_goal}"
    return StaticToolChoiceModel(
        (
            ModelSelectedToolCall(
                "artifact_write",
                {
                    "relative_path": "live_agent_turn/evidence.txt",
                    "content": content,
                },
            ),
        ),
        model_id=f"{handle.model.provider}/{handle.model.name}",
    )


def _uses_static_fallback(handle: AgentSessionHandle) -> bool:
    return handle.model.provider in {"offline", "static"}


def _endpoint(handle: AgentSessionHandle) -> ModelProviderConfig:
    provider = "local_gateway" if _uses_static_fallback(handle) else handle.model.provider
    return ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": handle.model.name,
            "reasoning_effort": handle.model.reasoning_effort,
            "base_url": handle.model.base_url,
            "auth_mode": handle.model.auth_mode,
            "api_key_env": handle.model.api_key_env,
        }
    )


def _ensure_turn_user_message(session_dir: Path, agent_id: str, user_goal: str) -> AgentSessionHandle:
    handle = load_agent_registry(session_dir).handles[agent_id]
    if _last_user_message_matches(handle, user_goal):
        return handle
    return append_agent_message(session_dir, agent_id, "user", user_goal)


def _last_user_message_matches(handle: AgentSessionHandle, user_goal: str) -> bool:
    records = read_jsonl(handle.messages_path)
    if not records:
        return False
    last = records[-1]
    return last.get("role") == "user" and last.get("content") == user_goal


def _chat_messages(handle: AgentSessionHandle) -> list[JsonMap]:
    records = read_jsonl(handle.messages_path)
    if records is None:
        return []
    messages: list[JsonMap] = []
    for record in records:
        role = record.get("role")
        content = record.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def _compact_summary(handle: AgentSessionHandle) -> str:
    payload = read_json(handle.session_dir / COMPACT_SUMMARY_NAME)
    if not payload:
        return ""
    if not _compact_summary_is_validated(payload):
        return ""
    summary = payload.get("summary")
    return summary if isinstance(summary, str) else ""


def _compact_summary_is_validated(payload: JsonMap) -> bool:
    compact_mode = payload.get("compact_mode")
    if compact_mode == "auto":
        return True
    return payload.get("manual_replay_status") == "passed"


def _assistant_message(status: str, blockers: tuple[str, ...]) -> str:
    if blockers:
        return f"agent loop blocked: {', '.join(blockers)}"
    return f"agent loop completed with status {status}"


def _safe_run_id(agent_id: str) -> str:
    return f"direct-{agent_id}-turn"


def _blocked_dispatch(
    request: SendAgentMessageRequest,
    blocker: str,
    bus_statuses: tuple[str, ...],
) -> LiveAgentDispatchResult:
    return LiveAgentDispatchResult(
        agent_id=request.to_agent,
        agent_session_id="",
        message_id=request.message_id,
        thread_id=request.thread_id,
        status="blocked",
        turn_status="blocked",
        model_id="",
        selected_tools=(),
        bus_statuses=bus_statuses,
        blockers=(blocker,),
    )
