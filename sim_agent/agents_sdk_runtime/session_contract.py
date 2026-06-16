from __future__ import annotations

from sim_agent.schemas._parse import JsonMap

from .runtime import AGENT_ROLES
from .session_events import INTER_AGENT_CALL_TIMEOUT_S, TEAM_HEARTBEAT_INTERVAL_S, call_matrix, role_ids


def agent_team_session_contract() -> JsonMap:
    role_id_set = role_ids(AGENT_ROLES)
    return {
        "contract_version": "agent_team_session_contract_v1",
        "heartbeat_interval_s": TEAM_HEARTBEAT_INTERVAL_S,
        "inter_agent_call_timeout_s": INTER_AGENT_CALL_TIMEOUT_S,
        "call_matrix": {agent: list(peers) for agent, peers in call_matrix(role_id_set).items()},
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
