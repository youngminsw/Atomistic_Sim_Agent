from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .runtime import AGENT_ROLES


AGENT_TEAM_SESSION_LEDGER_NAME = "agent_team_session_ledger.json"
TEAM_HEARTBEAT_INTERVAL_S = 3600
INTER_AGENT_CALL_TIMEOUT_S = 1800


class AgentTeamSessionStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


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


@dataclass(frozen=True, slots=True)
class AgentTeamSessionResult:
    session_id: str
    status: AgentTeamSessionStatus
    provider: str
    model: str
    auth_mode: str
    heartbeat_interval_s: int
    inter_agent_call_timeout_s: int
    call_matrix: dict[str, tuple[str, ...]]
    qa_gates: dict[str, str]
    session_files: tuple[str, ...]
    events: tuple[AgentTeamSessionEvent, ...]
    hard_blockers: tuple[str, ...]
    recoverable_events: tuple[str, ...]
    deadlock: bool

    @property
    def ok(self) -> bool:
        return not self.hard_blockers


def run_agent_team_session_smoke(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
    *,
    simulate_agent_failure: str | None = None,
    slurm_job_script: bool = False,
    qa_job_script_reviewed: bool = False,
) -> AgentTeamSessionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"agent-team-{_request_id(payload)}"
    role_ids = _role_ids()
    call_matrix = _call_matrix(role_ids)
    events = _bootstrap_events(session_id, role_ids)
    qa_gates: dict[str, str] = {"slurm_job_script": "not_required"}
    hard_blockers: list[str] = []
    recoverable_events: list[str] = []
    status = AgentTeamSessionStatus.READY

    events.extend(_exercise_bounded_calls(session_id, call_matrix))

    if simulate_agent_failure:
        if simulate_agent_failure not in role_ids:
            status = AgentTeamSessionStatus.BLOCKED
            hard_blockers.append(f"unknown_agent={simulate_agent_failure}")
        else:
            status = AgentTeamSessionStatus.DEGRADED
            recoverable_events.append(f"agent_failure:{simulate_agent_failure}")
            events.extend(_failure_recovery_events(session_id, simulate_agent_failure))

    if slurm_job_script:
        if qa_job_script_reviewed:
            qa_gates["slurm_job_script"] = "pass"
            events.append(
                _event(
                    session_id,
                    "qa_gate_pass",
                    "qa_agent",
                    "Slurm job script reviewed before compute submission",
                    "pass",
                    artifact_ref="qa/slurm_job_script_review",
                )
            )
        else:
            qa_gates["slurm_job_script"] = "required"
            status = AgentTeamSessionStatus.BLOCKED
            hard_blockers.append("qa_job_script_review_required")
            events.append(
                _event(
                    session_id,
                    "qa_gate_required",
                    "qa_agent",
                    "Slurm job script must pass QA before compute submission",
                    "blocked",
                    artifact_ref="qa/slurm_job_script_review",
                )
            )

    session_files = _write_session_files(output_dir, role_ids, events)
    result = AgentTeamSessionResult(
        session_id=session_id,
        status=status,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        heartbeat_interval_s=TEAM_HEARTBEAT_INTERVAL_S,
        inter_agent_call_timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        call_matrix=call_matrix,
        qa_gates=qa_gates,
        session_files=tuple(str(path) for path in session_files),
        events=tuple(events),
        hard_blockers=tuple(hard_blockers),
        recoverable_events=tuple(recoverable_events),
        deadlock=False,
    )
    write_agent_team_session_ledger(output_dir, result)
    return result


def write_agent_team_session_ledger(output_dir: Path, result: AgentTeamSessionResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / AGENT_TEAM_SESSION_LEDGER_NAME
    path.write_text(json.dumps(agent_team_session_payload(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def agent_team_session_payload(result: AgentTeamSessionResult) -> JsonMap:
    return {
        "ledger_version": "agent_team_session_v1",
        "session_id": result.session_id,
        "status": result.status.value,
        "provider": result.provider,
        "model": result.model,
        "auth_mode": result.auth_mode,
        "heartbeat_interval_s": result.heartbeat_interval_s,
        "inter_agent_call_timeout_s": result.inter_agent_call_timeout_s,
        "call_matrix": {agent: list(peers) for agent, peers in result.call_matrix.items()},
        "qa_gates": result.qa_gates,
        "session_files": list(result.session_files),
        "events": [asdict(event) for event in result.events],
        "hard_blockers": list(result.hard_blockers),
        "recoverable_events": list(result.recoverable_events),
        "deadlock": result.deadlock,
    }


def _role_ids() -> tuple[str, ...]:
    return ("orchestrator",) + tuple(role.role_id for role in AGENT_ROLES)


def _call_matrix(role_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    specialists = tuple(role for role in role_ids if role != "orchestrator")
    matrix: dict[str, tuple[str, ...]] = {"orchestrator": specialists}
    common_peers = ("orchestrator", "research_graphdb_agent", "qa_agent")
    for role_id in specialists:
        matrix[role_id] = tuple(peer for peer in common_peers if peer != role_id)
    return matrix


def _bootstrap_events(session_id: str, role_ids: tuple[str, ...]) -> list[AgentTeamSessionEvent]:
    events: list[AgentTeamSessionEvent] = []
    for role_id in role_ids:
        events.append(_event(session_id, "session_created", role_id, "durable session opened", "ready"))
        events.append(_event(session_id, "harness_ready", role_id, "role-local harness initialized", "ready"))
        events.append(
            _event(
                session_id,
                "heartbeat_registered",
                role_id,
                "long-running job heartbeat policy registered",
                "ready",
                heartbeat_interval_s=TEAM_HEARTBEAT_INTERVAL_S,
            )
        )
        events.append(
            _event(
                session_id,
                "context_compaction_checkpoint",
                role_id,
                "session can compact prior peer messages into durable summary",
                "ready",
                artifact_ref=f"sessions/{role_id}.jsonl",
            )
        )
    return events


def _exercise_bounded_calls(
    session_id: str,
    call_matrix: dict[str, tuple[str, ...]],
) -> list[AgentTeamSessionEvent]:
    events: list[AgentTeamSessionEvent] = []
    for agent_id, peers in call_matrix.items():
        for peer in peers:
            events.append(
                _event(
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


def _failure_recovery_events(session_id: str, failed_agent: str) -> list[AgentTeamSessionEvent]:
    return [
        _event(
            session_id,
            "agent_failure",
            failed_agent,
            "simulated recoverable worker failure",
            "failed",
            timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        ),
        _event(
            session_id,
            "recovery_route",
            "orchestrator",
            f"route {failed_agent} failure to QA and continue session supervision",
            "degraded",
            peer="qa_agent",
            timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        ),
        _event(
            session_id,
            "qa_recovery_review",
            "qa_agent",
            f"review recoverable {failed_agent} failure without deadlock",
            "degraded",
            peer="orchestrator",
        ),
    ]


def _write_session_files(
    output_dir: Path,
    role_ids: tuple[str, ...],
    events: list[AgentTeamSessionEvent],
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


def _event(
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


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"
