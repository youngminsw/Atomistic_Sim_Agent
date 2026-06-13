from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES
from sim_agent.agents_sdk_runtime.session_contract import agent_team_session_contract
from sim_agent.agents_sdk_runtime.session_runtime import (
    AGENT_TEAM_SESSION_LEDGER_NAME,
    AgentTeamSessionEvent,
    AgentTeamSessionResult,
    TEAM_HEARTBEAT_INTERVAL_S,
    run_agent_team_session_smoke,
)
from sim_agent.llm_endpoints import ModelProviderConfig

from .tui_catalog import SIMULATION_SKILLS
from .tui_parse import parse_options
from .tui_render import AgentStatusRow, write_agent_workboard
from .tui_state import TuiState, append_event, replace_team_ledger

SHORT_BOUNDARIES: dict[str, str] = {
    "md_agent": "LAMMPS MD and physics gates",
    "ml_mdn_agent": "MDN training and uncertainty gate",
    "feature_scale_agent": "KMC and Level-Set evolution",
    "research_graphdb_agent": "GraphDB research and provenance",
    "qa_agent": "QA evidence and blocker audit",
}


def handle_agents(output_stream: TextIO) -> None:
    write_agent_workboard("Agent Workboard", _roster_rows(), output_stream)
    output_stream.write("agent_roster=true\n")
    output_stream.write("agent=orchestrator boundary=routes work, approvals, and final run assembly\n")
    for role in AGENT_ROLES:
        output_stream.write(f"agent={role.role_id} boundary={SHORT_BOUNDARIES[role.role_id]}\n")
        output_stream.write(
            f"agent_activity={role.role_id} ready "
            f"summary=role-local harness initialized heartbeat {TEAM_HEARTBEAT_INTERVAL_S}s\n"
        )


def handle_skills(output_stream: TextIO) -> None:
    output_stream.write("skill_catalog=true\n")
    for name, summary in SIMULATION_SKILLS:
        output_stream.write(f"skill={name} summary={summary}\n")


def handle_harness(output_stream: TextIO) -> None:
    contract = agent_team_session_contract()
    output_stream.write("Harness Contract\n")
    output_stream.write("harness_contract=true\n")
    output_stream.write(f"heartbeat_interval_s={contract['heartbeat_interval_s']}\n")
    output_stream.write(f"inter_agent_call_timeout_s={contract['inter_agent_call_timeout_s']}\n")
    output_stream.write(f"failure_policy={contract['failure_policy']}\n")
    _write_call_matrix(contract["call_matrix"], output_stream)
    _write_qa_gates(contract["qa_gates"], output_stream)
    artifacts = contract["session_artifacts"]
    if isinstance(artifacts, list):
        for artifact in artifacts:
            output_stream.write(f"session_artifact={artifact}\n")


def handle_team(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": state.model.provider,
            "model": state.model.name,
            "reasoning_effort": "high",
            "base_url": state.model.base_url,
            "auth_mode": state.model.auth_mode,
            "api_key_env": state.model.api_key_env,
        }
    )
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "team")))
    result = run_agent_team_session_smoke(
        {"request_id": state.session_id},
        endpoint,
        output_dir,
        simulate_agent_failure=parsed.options.get("simulate_agent_failure"),
        slurm_job_script="slurm_job_script" in parsed.flags,
        qa_job_script_reviewed="qa_reviewed" in parsed.flags,
    )
    ledger = output_dir / AGENT_TEAM_SESSION_LEDGER_NAME
    next_state = replace_team_ledger(state, ledger)
    append_event(next_state, "team_session", f"{result.session_id}:{result.status.value}")
    output_stream.write("team_session_ready=true\n")
    output_stream.write(f"team_session_id={result.session_id}\n")
    output_stream.write(f"team_status={result.status.value}\n")
    output_stream.write(f"heartbeat_interval_s={result.heartbeat_interval_s}\n")
    output_stream.write(f"inter_agent_call_timeout_s={result.inter_agent_call_timeout_s}\n")
    output_stream.write(f"team_ledger_path={ledger}\n")
    write_agent_workboard("Agent Workboard", _result_rows(result), output_stream)
    if result.hard_blockers:
        output_stream.write(f"hard_blockers={','.join(result.hard_blockers)}\n")
    return next_state


def handle_contract(output_stream: TextIO) -> None:
    contract = agent_team_session_contract()
    output_stream.write("team_contract=true\n")
    output_stream.write(f"heartbeat_interval_s={contract['heartbeat_interval_s']}\n")
    output_stream.write(f"inter_agent_call_timeout_s={contract['inter_agent_call_timeout_s']}\n")
    output_stream.write(f"failure_policy={contract['failure_policy']}\n")


def _roster_rows() -> tuple[AgentStatusRow, ...]:
    rows = [AgentStatusRow("orchestrator", "ready", "routes work, approvals, and final run assembly")]
    rows.extend(
        AgentStatusRow(role.role_id, "ready", "role-local harness initialized", heartbeat_s=TEAM_HEARTBEAT_INTERVAL_S)
        for role in AGENT_ROLES
    )
    return tuple(rows)


def _result_rows(result: AgentTeamSessionResult) -> tuple[AgentStatusRow, ...]:
    latest = _latest_events(result.events)
    rows: list[AgentStatusRow] = []
    for agent_id in result.call_matrix:
        event = latest[agent_id]
        rows.append(
            AgentStatusRow(
                agent_id,
                event.status,
                event.summary,
                peer=event.peer or "",
                heartbeat_s=result.heartbeat_interval_s,
            )
        )
    return tuple(rows)


def _write_call_matrix(value, output_stream: TextIO) -> None:
    if not isinstance(value, dict):
        return
    for agent_id, peers in sorted(value.items()):
        if not isinstance(agent_id, str) or not isinstance(peers, list):
            continue
        peer_text = ",".join(peer for peer in peers if isinstance(peer, str))
        output_stream.write(f"call={agent_id}->{peer_text}\n")


def _write_qa_gates(value, output_stream: TextIO) -> None:
    if not isinstance(value, dict):
        return
    for gate_id, policy in sorted(value.items()):
        if isinstance(gate_id, str) and isinstance(policy, str):
            output_stream.write(f"qa_gate={gate_id}:{policy}\n")


def _latest_events(events: tuple[AgentTeamSessionEvent, ...]) -> dict[str, AgentTeamSessionEvent]:
    latest: dict[str, AgentTeamSessionEvent] = {}
    for event in events:
        latest[event.agent_id] = event
    return latest
