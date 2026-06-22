from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES
from sim_agent.schemas._parse import JsonMap

from .tui_render import AgentStatusRow
from .tui_state import SESSION_EVENTS_NAME, TuiState


DIRECT_SESSION_ACTIVITY: Final = "persistent agent session accepted message"
IDLE_ORCHESTRATOR_ACTIVITY: Final = "routes work, approvals, and final run assembly"
IDLE_SPECIALIST_ACTIVITY: Final = "role-local harness initialized"


@dataclass(frozen=True, slots=True)
class AgentActivitySummary:
    rows: tuple[AgentStatusRow, ...]
    active_agent: str
    mode: str


def build_agent_activity_summary(state: TuiState, *, heartbeat_s: int | None = None) -> AgentActivitySummary:
    active_agent = _latest_direct_agent(state)
    mode = "direct_session" if active_agent and active_agent != "orchestrator" else "idle"
    rows: list[AgentStatusRow] = [
        AgentStatusRow(
            "orchestrator",
            "ready",
            f"direct @{active_agent} session active" if active_agent else IDLE_ORCHESTRATOR_ACTIVITY,
        )
    ]
    for role in AGENT_ROLES:
        if role.role_id == active_agent:
            rows.append(AgentStatusRow(role.role_id, "direct_session", DIRECT_SESSION_ACTIVITY, heartbeat_s=heartbeat_s))
            continue
        rows.append(AgentStatusRow(role.role_id, "ready", IDLE_SPECIALIST_ACTIVITY, heartbeat_s=heartbeat_s))
    return AgentActivitySummary(tuple(rows), active_agent or "-", mode)


def agent_activity_hud_label(state: TuiState) -> str:
    summary = build_agent_activity_summary(state)
    if summary.mode == "direct_session":
        return f"{summary.active_agent} · direct-session · persistent handle"
    return "no active specialist session"


def _latest_direct_agent(state: TuiState) -> str:
    for event in reversed(_safe_session_events(state, limit=64)):
        if event.get("event_type") != "agent_direct_route":
            continue
        summary = event.get("summary")
        if isinstance(summary, str) and summary:
            return summary
    return ""


def _safe_session_events(state: TuiState, *, limit: int) -> tuple[JsonMap, ...]:
    path = state.session_dir / SESSION_EVENTS_NAME
    if not path.is_file():
        return ()
    events: list[JsonMap] = []
    for line in path.read_bytes().splitlines()[-limit:]:
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return tuple(events)
