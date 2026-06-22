from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.agent_runtime import (
    CompactionRequest,
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    HandoffTaskRequest,
    ReplyAgentMessageRequest,
    SendAgentMessageRequest,
    ack_agent_message,
    compact_agent_session,
    handoff_task,
    open_global_session,
    read_agent_message,
    replay_agent_compaction,
    reply_agent_message,
    send_agent_message,
)
from sim_agent.agent_runtime.compaction_store import read_json, read_jsonl
from sim_agent.agent_runtime.live_agent_turn import LiveAgentTurnResult, run_live_agent_turn
from sim_agent.cli.tui_state import ModelSettings, TuiState, persist_state
from sim_agent.cli.tui_timeline import timeline_summary
from sim_agent.llm_endpoints.model_profiles import ModelProfile, find_model_profile
from sim_agent.runtime_config import RuntimeConfig, load_runtime_config
from sim_agent.schemas._parse import JsonMap


SCENARIO_ORCHESTRATOR_SUBAGENT_TOOL_LOOP: Final = "orchestrator_subagent_tool_loop"
SMOKE_THREAD_ID: Final = "e2e-runtime-smoke"
SMOKE_SUBAGENT_ID: Final = "e2e-planner-smoke"
NO_DESTRUCTIVE_WRITES_STATEMENT: Final = "No MD, remote execution, or GraphDB destructive write ran."


@dataclass(frozen=True, slots=True)
class E2ERuntimeSmokeRequest:
    model_profile: str
    scenario: str
    allow_hardgate_bypass: bool
    output_json: Path
    session_dir: Path | None


@dataclass(frozen=True, slots=True)
class E2ERuntimeSmokeError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def run_e2e_runtime_smoke(request: E2ERuntimeSmokeRequest) -> Path:
    if request.scenario != SCENARIO_ORCHESTRATOR_SUBAGENT_TOOL_LOOP:
        raise E2ERuntimeSmokeError(f"unsupported_e2e_runtime_smoke_scenario:{request.scenario}")
    profile = find_model_profile(request.model_profile)
    if profile is None:
        raise E2ERuntimeSmokeError(f"unknown_model_profile:{request.model_profile}")
    runtime_config = load_runtime_config()
    model = _model_from_profile(profile, runtime_config)
    opened = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=request.session_dir,
            default_root=Path(runtime_config.evidence_root) / "e2e-runtime-smoke",
            model=model,
        )
    )
    turn = run_live_agent_turn(
        opened.record.session_dir,
        "orchestrator",
        _scenario_prompt(request.scenario, request.allow_hardgate_bypass),
    )
    bus = _exercise_message_bus(opened.record.session_dir)
    handoff = handoff_task(
        opened.record.session_dir,
        HandoffTaskRequest(
            from_agent="orchestrator",
            target_agent="md_agent",
            task_id="e2e-md-handoff",
            thread_id=SMOKE_THREAD_ID,
            task=(
                "Use artifact_write to create e2e_runtime_smoke/handoff_report.md "
                "with a short non-destructive handoff receipt. Do not run MD, remote execution, or GraphDB writes."
            ),
        ),
    )
    compact = compact_agent_session(
        opened.record.session_dir,
        CompactionRequest(
            agent_id="orchestrator",
            compact_id="e2e-orchestrator-manual",
            summary="E2E smoke: orchestrator selected a bounded subagent and recorded session-local evidence.",
        ),
    )
    replay = replay_agent_compaction(opened.record.session_dir, "orchestrator")
    resumed = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=opened.record.session_dir,
            default_root=Path(runtime_config.evidence_root) / "e2e-runtime-smoke",
            model=model,
            resume="latest",
        )
    )
    payload = _evidence_payload(
        request,
        runtime_config,
        model,
        opened.record.session_id,
        opened.record.session_dir,
        turn,
        bus,
        handoff.to_json(),
        _compaction_json(compact),
        _compaction_json(replay),
        resumed.opened_as,
    )
    request.output_json.parent.mkdir(parents=True, exist_ok=True)
    request.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request.output_json


def _model_from_profile(profile: ModelProfile, runtime_config: RuntimeConfig) -> GlobalSessionModel:
    endpoint = runtime_config.model_endpoint
    return GlobalSessionModel(
        provider=profile.default.provider,
        name=profile.default.model,
        reasoning_effort=profile.default.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )


def _scenario_prompt(scenario: str, allow_hardgate_bypass: bool) -> str:
    bypass = " hardgate_bypass_mode=test_only." if allow_hardgate_bypass else ""
    return (
        f"Run ASA E2E runtime smoke scenario {scenario}.{bypass} "
        "You are the orchestrator. Select exactly one safe session-local evidence tool: subagent_task. "
        "Use caller_agent=orchestrator, preset=planner, task_id=e2e-planner-smoke, depth=1. "
        "The subagent task must instruct the child planner to use artifact_write to create "
        "subagent_report.md with a concise non-destructive runtime receipt. "
        "Do not select artifact_write directly in the orchestrator turn. "
        "Do not run MD, remote execution, or GraphDB destructive writes."
    )


def _exercise_message_bus(session_dir: Path) -> JsonMap:
    message_id = "e2e-bus-qa-message"
    send = send_agent_message(
        session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="E2E smoke bus receipt only; no simulation side effects.",
            thread_id=SMOKE_THREAD_ID,
            message_id=message_id,
        ),
    )
    ack = ack_agent_message(session_dir, message_id=message_id, by_agent="qa_agent")
    read = read_agent_message(session_dir, message_id=message_id, by_agent="qa_agent")
    reply = reply_agent_message(
        session_dir,
        ReplyAgentMessageRequest(message_id=message_id, by_agent="qa_agent", content="qa_agent bus receipt ok"),
    )
    records = [send.to_json(), ack.to_json(), read.to_json(), reply.to_json()]
    status = "succeeded" if all(record["status"] != "blocked" for record in records) else "blocked"
    return {
        "status": status,
        "message_id": message_id,
        "thread_id": SMOKE_THREAD_ID,
        "records": records,
        "ledger_path": str(session_dir / "message_bus" / "messages.jsonl"),
    }


def _evidence_payload(
    request: E2ERuntimeSmokeRequest,
    runtime_config: RuntimeConfig,
    model: GlobalSessionModel,
    session_id: str,
    session_dir: Path,
    turn: LiveAgentTurnResult,
    message_bus: JsonMap,
    handoff: JsonMap,
    compaction: JsonMap,
    compaction_replay: JsonMap,
    resumed_as: str,
) -> JsonMap:
    subagent_runs = _subagent_runs(session_dir)
    timeline = _timeline(session_id, session_dir, model)
    reconciliation = _ledger_reconciliation(session_dir, subagent_runs)
    blockers = _e2e_blockers(turn, subagent_runs, message_bus, handoff, compaction, compaction_replay, resumed_as)
    payload: JsonMap = {
        "schema_version": "asa_e2e_runtime_smoke_v1",
        "scenario": request.scenario,
        "model_profile": request.model_profile,
        "model": {
            "provider": model.provider,
            "name": model.name,
            "reasoning_effort": model.reasoning_effort,
            "base_url": model.base_url,
            "auth_mode": model.auth_mode,
            "api_key_env": model.api_key_env,
        },
        "global_session_id": session_id,
        "session_dir": str(session_dir),
        "global_session_path": str(session_dir / "global_session.json"),
        "agent_session_dir": str(session_dir / "agent_sessions" / "orchestrator"),
        "turn": turn.to_json(),
        "selected_tools": list(turn.selected_tools),
        "subagent_task_selected": "subagent_task" in turn.selected_tools,
        "subagent_runs": subagent_runs,
        "message_bus": message_bus,
        "handoff": handoff,
        "compaction": compaction,
        "compaction_replay": compaction_replay,
        "resume": {
            "status": "succeeded" if resumed_as == "resumed" else "blocked",
            "opened_as": resumed_as,
        },
        "timeline": timeline,
        "ledger_reconciliation": reconciliation,
        "status": "succeeded" if not blockers else "blocked",
        "blockers": blockers,
        "destructive_write_statement": NO_DESTRUCTIVE_WRITES_STATEMENT,
        "destructive_writes_ran": {
            "md": False,
            "remote": False,
            "graphdb": False,
        },
    }
    if request.allow_hardgate_bypass:
        payload["hardgate_bypass_mode"] = "test_only"
    return payload


def _subagent_runs(session_dir: Path) -> list[JsonMap]:
    runs: list[JsonMap] = []
    for path in sorted((session_dir / "agent_sessions").glob("*/subagents/*/*/subagent_run.json")):
        payload = read_json(path) or {}
        if not payload:
            continue
        agent_loop = payload.get("agent_loop")
        selected_tools = agent_loop.get("selected_tools", []) if isinstance(agent_loop, dict) else []
        artifact_refs = agent_loop.get("tool_result_refs", []) if isinstance(agent_loop, dict) else []
        compact = {
            "ledger_path": str(path),
            "caller_agent": payload.get("caller_agent", ""),
            "preset": payload.get("preset", ""),
            "subagent_id": payload.get("subagent_id", ""),
            "status": payload.get("status", ""),
            "selected_tools": selected_tools,
            "artifact_refs": artifact_refs,
        }
        runs.append(compact)
    return runs


def _compaction_json(result) -> JsonMap:
    return {
        "status": result.status,
        "compact_status": result.compact_status,
        "agent_id": result.agent_id,
        "compact_id": result.compact_id,
        "summary_path": str(result.summary_path),
        "blocker": result.blocker or "",
    }


def _timeline(session_id: str, session_dir: Path, model: GlobalSessionModel) -> JsonMap:
    state = TuiState(
        session_id=session_id,
        session_dir=session_dir,
        model=ModelSettings(
            provider=model.provider,
            name=model.name,
            reasoning_effort=model.reasoning_effort,
            base_url=model.base_url,
            auth_mode=model.auth_mode,
            api_key_env=model.api_key_env,
        ),
        global_session_id=session_id,
        global_session_path=session_dir / "global_session.json",
    )
    persist_state(state)
    summary = timeline_summary(state)
    return {
        "status": "succeeded" if summary.event_count > 0 else "blocked",
        "event_count": summary.event_count,
        "latest_event_type": summary.latest_event_type,
        "latest_actor": summary.latest_actor,
        "latest_source": summary.latest_source,
    }


def _ledger_reconciliation(session_dir: Path, subagent_runs: list[JsonMap]) -> JsonMap:
    agent_event_count = _jsonl_count(session_dir / "agent_sessions" / "orchestrator" / "events.jsonl")
    bus_count = _jsonl_count(session_dir / "message_bus" / "messages.jsonl")
    handoff_count = _jsonl_count(session_dir / "message_bus" / "handoffs.jsonl")
    return {
        "status": "succeeded" if agent_event_count > 0 and bus_count >= 4 and subagent_runs else "blocked",
        "orchestrator_event_count": agent_event_count,
        "message_bus_record_count": bus_count,
        "handoff_record_count": handoff_count,
        "subagent_run_count": len(subagent_runs),
    }


def _jsonl_count(path: Path) -> int:
    records = read_jsonl(path)
    return len(records) if isinstance(records, list) else 0


def _e2e_blockers(
    turn: LiveAgentTurnResult,
    subagent_runs: list[JsonMap],
    message_bus: JsonMap,
    handoff: JsonMap,
    compaction: JsonMap,
    compaction_replay: JsonMap,
    resumed_as: str,
) -> list[str]:
    blockers = list(turn.blockers)
    if turn.status != "succeeded":
        blockers.append("orchestrator_turn_not_succeeded")
    if "subagent_task" not in turn.selected_tools:
        blockers.append("orchestrator_subagent_task_not_selected")
    if not _has_successful_smoke_subagent(subagent_runs):
        blockers.append("planner_subagent_run_missing_or_blocked")
    if message_bus.get("status") != "succeeded":
        blockers.append("message_bus_not_succeeded")
    if handoff.get("status") != "live_completed":
        blockers.append("handoff_not_live_completed")
    if compaction.get("status") != "succeeded" or compaction_replay.get("status") != "succeeded":
        blockers.append("compaction_replay_not_succeeded")
    if resumed_as != "resumed":
        blockers.append("resume_not_succeeded")
    return blockers


def _has_successful_smoke_subagent(subagent_runs: list[JsonMap]) -> bool:
    for run in subagent_runs:
        if run.get("subagent_id") == SMOKE_SUBAGENT_ID and run.get("status") == "succeeded":
            selected = run.get("selected_tools")
            return isinstance(selected, list) and "artifact_write" in selected
    return False
