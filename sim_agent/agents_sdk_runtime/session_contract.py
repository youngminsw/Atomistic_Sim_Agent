from __future__ import annotations

from sim_agent.schemas._parse import JsonMap

from .runtime import AGENT_ROLES
from .session_runtime import INTER_AGENT_CALL_TIMEOUT_S, TEAM_HEARTBEAT_INTERVAL_S


def agent_team_session_contract() -> JsonMap:
    role_ids = _role_ids()
    return {
        "contract_version": "agent_team_session_contract_v1",
        "heartbeat_interval_s": TEAM_HEARTBEAT_INTERVAL_S,
        "inter_agent_call_timeout_s": INTER_AGENT_CALL_TIMEOUT_S,
        "call_matrix": {agent: list(peers) for agent, peers in _call_matrix(role_ids).items()},
        "qa_gates": {
            "slurm_job_script": "qa_before_submit",
            "long_compute_submission": "qa_before_submit",
        },
        "failure_policy": "owning_agent_recovers_then_routes_to_orchestrator_and_qa",
        "session_artifacts": [
            "agent_team_session_ledger.json",
            "sessions/<agent_id>.jsonl",
        ],
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
