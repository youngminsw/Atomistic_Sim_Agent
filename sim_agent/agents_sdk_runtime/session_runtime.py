from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import JsonMap

from .graph_memory import runtime_graph_memory_payload
from .runtime import AGENT_ROLES
from .session_events import (
    INTER_AGENT_CALL_TIMEOUT_S,
    TEAM_HEARTBEAT_INTERVAL_S,
    AgentTeamSessionEvent,
    bootstrap_events,
    bounded_call_events,
    call_matrix,
    failure_recovery_events,
    role_ids,
    session_event,
    write_session_files,
)


class AgentTeamSessionStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


AGENT_TEAM_SESSION_LEDGER_NAME = "agent_team_session_ledger.json"


@dataclass(frozen=True, slots=True)
class AgentTeamSessionResult:
    session_id: str
    execution_mode: str
    status: AgentTeamSessionStatus
    provider: str
    model: str
    auth_mode: str
    heartbeat_interval_s: int
    inter_agent_call_timeout_s: int
    call_matrix: dict[str, tuple[str, ...]]
    qa_gates: dict[str, str]
    graph_memory: JsonMap
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
    return _run_agent_team_session(
        payload,
        endpoint,
        output_dir,
        execution_mode="smoke",
        simulate_agent_failure=simulate_agent_failure,
        slurm_job_script=slurm_job_script,
        qa_job_script_reviewed=qa_job_script_reviewed,
    )


def run_agent_team_session_runtime(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
) -> AgentTeamSessionResult:
    return _run_agent_team_session(
        payload,
        endpoint,
        output_dir,
        execution_mode="team_contract_runtime",
        runtime_primary=True,
    )


def _run_agent_team_session(
    payload: JsonMap,
    endpoint: ModelProviderConfig,
    output_dir: Path,
    *,
    execution_mode: str,
    runtime_primary: bool = False,
    simulate_agent_failure: str | None = None,
    slurm_job_script: bool = False,
    qa_job_script_reviewed: bool = False,
) -> AgentTeamSessionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"agent-team-{_request_id(payload)}"
    role_id_set = role_ids(AGENT_ROLES)
    team_call_matrix = call_matrix(role_id_set)
    events = bootstrap_events(session_id, role_id_set)
    graph_memory = runtime_graph_memory_payload(payload, role_id_set)
    events.extend(_graph_memory_events(session_id, role_id_set))
    qa_gates: dict[str, str] = {"slurm_job_script": "not_required"}
    hard_blockers: list[str] = []
    recoverable_events: list[str] = []
    status = AgentTeamSessionStatus.READY

    events.extend(bounded_call_events(session_id, team_call_matrix))
    if runtime_primary:
        events.append(
            session_event(
                session_id,
                "team_runtime_primary",
                "orchestrator",
                "TUI default goal entered team contract runtime before bundle preparation",
                "ready",
                artifact_ref="team_contract_runtime",
            )
        )
        events.append(
            session_event(
                session_id,
                "team_contract_runtime_ready",
                "qa_agent",
                "bounded team call matrix, QA gates, recovery policy, and session ledgers are ready",
                "ready",
                artifact_ref="agent_team_session_ledger.json",
            )
        )

    if simulate_agent_failure:
        if simulate_agent_failure not in role_id_set:
            status = AgentTeamSessionStatus.BLOCKED
            hard_blockers.append(f"unknown_agent={simulate_agent_failure}")
        else:
            status = AgentTeamSessionStatus.DEGRADED
            recoverable_events.append(f"agent_failure:{simulate_agent_failure}")
            events.extend(failure_recovery_events(session_id, simulate_agent_failure))

    if slurm_job_script:
        if qa_job_script_reviewed:
            qa_gates["slurm_job_script"] = "pass"
            events.append(
                session_event(
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
                session_event(
                    session_id,
                    "qa_gate_required",
                    "qa_agent",
                    "Slurm job script must pass QA before compute submission",
                    "blocked",
                    artifact_ref="qa/slurm_job_script_review",
                )
            )

    session_files = write_session_files(output_dir, role_id_set, events)
    result = AgentTeamSessionResult(
        session_id=session_id,
        execution_mode=execution_mode,
        status=status,
        provider=endpoint.provider,
        model=endpoint.model,
        auth_mode=endpoint.auth_mode,
        heartbeat_interval_s=TEAM_HEARTBEAT_INTERVAL_S,
        inter_agent_call_timeout_s=INTER_AGENT_CALL_TIMEOUT_S,
        call_matrix=team_call_matrix,
        qa_gates=qa_gates,
        graph_memory=graph_memory,
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
        "execution_mode": result.execution_mode,
        "status": result.status.value,
        "provider": result.provider,
        "model": result.model,
        "auth_mode": result.auth_mode,
        "heartbeat_interval_s": result.heartbeat_interval_s,
        "inter_agent_call_timeout_s": result.inter_agent_call_timeout_s,
        "call_matrix": {agent: list(peers) for agent, peers in result.call_matrix.items()},
        "qa_gates": result.qa_gates,
        "graph_memory": result.graph_memory,
        "session_files": list(result.session_files),
        "events": [asdict(event) for event in result.events],
        "hard_blockers": list(result.hard_blockers),
        "recoverable_events": list(result.recoverable_events),
        "deadlock": result.deadlock,
    }


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def _graph_memory_events(session_id: str, role_id_set: tuple[str, ...]) -> list[AgentTeamSessionEvent]:
    return [
        session_event(
            session_id,
            "graph_memory_context_attached",
            role_id,
            "GraphDB brain query context attached before agent work",
            "ready",
            artifact_ref="agent_graph_context",
        )
        for role_id in role_id_set
    ]
