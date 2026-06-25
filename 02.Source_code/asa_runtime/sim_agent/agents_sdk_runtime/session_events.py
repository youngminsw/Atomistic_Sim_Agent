from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import AgentRoleDefinition


TEAM_HEARTBEAT_INTERVAL_S = 3600
INTER_AGENT_CALL_TIMEOUT_S = 1800


@dataclass(frozen=True, slots=True)
class AgentTeamSessionEvent:
    at: float
    session_id: str
    event_type: str
    agent_id: str
    summary: str
    status: str
    peer: str | None = None
    timeout_s: int | None = None
    heartbeat_interval_s: int | None = None
    artifact_ref: str | None = None


def role_ids(agent_roles: tuple[AgentRoleDefinition, ...]) -> tuple[str, ...]:
    return ("orchestrator",) + tuple(role.role_id for role in agent_roles)


def call_matrix(role_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    specialists = tuple(role for role in role_ids if role != "orchestrator")
    matrix: dict[str, tuple[str, ...]] = {"orchestrator": specialists}
    common_peers = ("orchestrator", "research_agent", "qa_agent")
    for role_id in specialists:
        matrix[role_id] = tuple(peer for peer in common_peers if peer != role_id)
    return matrix


def bootstrap_events(session_id: str, role_ids: tuple[str, ...]) -> list[AgentTeamSessionEvent]:
    events: list[AgentTeamSessionEvent] = []
    for role_id in role_ids:
        events.append(session_event(session_id, "session_created", role_id, "durable session opened", "ready"))
        events.append(session_event(session_id, "harness_ready", role_id, "role-local harness initialized", "ready"))
        events.append(
            session_event(
                session_id,
                "heartbeat_registered",
                role_id,
                "long-running job heartbeat policy registered",
                "ready",
                heartbeat_interval_s=TEAM_HEARTBEAT_INTERVAL_S,
            )
        )
        events.append(
            session_event(
                session_id,
                "context_compaction_checkpoint",
                role_id,
                "session can compact prior peer messages into durable summary",
                "ready",
                artifact_ref=f"sessions/{role_id}.jsonl",
            )
        )
    return events


def bounded_call_events(
    session_id: str,
    call_matrix: dict[str, tuple[str, ...]],
) -> list[AgentTeamSessionEvent]:
    events: list[AgentTeamSessionEvent] = []
    for agent_id, peers in call_matrix.items():
        for peer in peers:
            events.append(
                session_event(
                    session_id,
                    "bounded_inter_agent_call",
                    agent_id,
                    f"call {peer} with finite wait policy",
                    "ack",
                    peer=peer,
                    timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
                )
            )
    return events


def failure_recovery_events(session_id: str, failed_agent: str) -> list[AgentTeamSessionEvent]:
    return [
        session_event(
            session_id,
            "agent_failure",
            failed_agent,
            "simulated recoverable worker failure",
            "failed",
            timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        ),
        session_event(
            session_id,
            "recovery_route",
            "orchestrator",
            f"route {failed_agent} failure to QA and continue session supervision",
            "degraded",
            peer="qa_agent",
            timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        ),
        session_event(
            session_id,
            "qa_recovery_review",
            "qa_agent",
            f"review recoverable {failed_agent} failure without deadlock",
            "degraded",
            peer="orchestrator",
        ),
    ]


def write_session_files(
    output_dir: Path,
    role_ids: tuple[str, ...],
    events: Sequence[AgentTeamSessionEvent],
) -> tuple[Path, ...]:
    session_dir = output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for role_id in role_ids:
        path = session_dir / f"{role_id}.jsonl"
        role_events = [event for event in events if event.agent_id == role_id]
        path.write_text(
            "".join(json.dumps(asdict(event), sort_keys=True) + "\n" for event in role_events),
            encoding="utf-8",
        )
        files.append(path)
    return tuple(files)


def session_event(
    session_id: str,
    event_type: str,
    agent_id: str,
    summary: str,
    status: str,
    *,
    peer: str | None = None,
    timeout_s: int | None = None,
    heartbeat_interval_s: int | None = None,
    artifact_ref: str | None = None,
) -> AgentTeamSessionEvent:
    return AgentTeamSessionEvent(
        at=time.time(),
        session_id=session_id,
        event_type=event_type,
        agent_id=agent_id,
        summary=summary,
        status=status,
        peer=peer,
        timeout_s=timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
        artifact_ref=artifact_ref,
    )
