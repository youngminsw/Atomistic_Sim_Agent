from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap


WORKFLOW_HARNESS_LEDGER_NAME: Final = "workflow_harness_ledger.json"
WORKFLOW_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    workflow_id: str
    display_name: str
    states: tuple[str, ...]
    current_state: str
    verification_gate: str
    required_evidence: tuple[str, ...]
    hook: str
    loop_policy: str


@dataclass(frozen=True, slots=True)
class WorkflowHarnessEvent:
    at: float
    workflow_id: str
    state: str
    hook: str
    summary: str
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class WorkflowHarnessResult:
    workflow_id: str
    status: str
    current_state: str
    verification_gate: str
    gate_status: str
    evidence_keys: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    resumable: bool
    ledger_ref: str
    blockers: tuple[str, ...]
    events: tuple[WorkflowHarnessEvent, ...]


WORKFLOW_DEFINITIONS: Final[tuple[WorkflowDefinition, ...]] = (
    WorkflowDefinition(
        "deep-interview",
        "Deep Interview",
        ("initialized", "question_round", "ambiguity_gate", "handoff_ready"),
        "handoff_ready",
        "ambiguity_gate_clear",
        ("question_answer", "ambiguity_score"),
        "UserPromptSubmit",
        "ask_one_round_then_checkpoint",
    ),
    WorkflowDefinition(
        "ralplan",
        "RALPlan",
        ("initialized", "requirements_loaded", "consensus_plan", "verification_plan_ready"),
        "verification_plan_ready",
        "plan_has_acceptance_tests",
        ("prd_path", "test_spec_path"),
        "SlashCommand",
        "plan_review_checkpoint",
    ),
    WorkflowDefinition(
        "ultrawork",
        "Ultrawork",
        ("initialized", "task_decomposition", "parallel_lanes_ready", "merge_ready"),
        "merge_ready",
        "lane_outputs_have_evidence",
        ("lane_outputs",),
        "SlashCommand",
        "parallel_lanes_with_merge_gate",
    ),
    WorkflowDefinition(
        "ultraqa",
        "UltraQA",
        ("initialized", "hostile_scenarios", "adversarial_checks", "fix_or_report_ready"),
        "fix_or_report_ready",
        "adversarial_scenarios_recorded",
        ("adversarial_scenarios",),
        "PostToolUse",
        "qa_loop_until_blockers_clear",
    ),
    WorkflowDefinition(
        "ultragoal",
        "Ultragoal",
        ("initialized", "goals_loaded", "active_story_resumed", "checkpoint_ready"),
        "checkpoint_ready",
        "codex_snapshot_reconciled_or_blocked",
        ("codex_goal_snapshot",),
        "GoalState",
        "story_checkpoint_after_verification",
    ),
)


def workflow_harness_catalog() -> tuple[WorkflowDefinition, ...]:
    return WORKFLOW_DEFINITIONS


def run_workflow_harness_smoke(workflow_id: str, payload: JsonMap, output_dir: Path) -> WorkflowHarnessResult:
    workflow = _workflow_definition(workflow_id)
    if workflow is None:
        return _blocked_result(workflow_id, output_dir, "unknown_workflow")
    workflow_dir = output_dir / workflow.workflow_id
    ledger_ref = f"{workflow.workflow_id}/{WORKFLOW_HARNESS_LEDGER_NAME}"
    evidence_keys = _evidence_keys(payload)
    missing = tuple(key for key in workflow.required_evidence if key not in evidence_keys)
    if missing:
        events = (
            _event(workflow, "initialized", False),
            WorkflowHarnessEvent(
                time.time(),
                workflow.workflow_id,
                "blocked",
                workflow.hook,
                f"{workflow.verification_gate}:missing_evidence={','.join(missing)}",
                terminal=True,
            ),
        )
        result = WorkflowHarnessResult(
            workflow_id=workflow.workflow_id,
            status="blocked",
            current_state="blocked",
            verification_gate=workflow.verification_gate,
            gate_status="blocked",
            evidence_keys=evidence_keys,
            missing_evidence=missing,
            resumable=True,
            ledger_ref=ledger_ref,
            blockers=("workflow_gate_missing_evidence",),
            events=events,
        )
        workflow_dir.mkdir(parents=True, exist_ok=True)
        (workflow_dir / WORKFLOW_HARNESS_LEDGER_NAME).write_text(
            json.dumps(_ledger_payload(result, workflow, payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    events = tuple(_event(workflow, state, state == workflow.current_state) for state in workflow.states)
    result = WorkflowHarnessResult(
        workflow_id=workflow.workflow_id,
        status="ready",
        current_state=workflow.current_state,
        verification_gate=workflow.verification_gate,
        gate_status="passed",
        evidence_keys=evidence_keys,
        missing_evidence=(),
        resumable=True,
        ledger_ref=ledger_ref,
        blockers=(),
        events=events,
    )
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / WORKFLOW_HARNESS_LEDGER_NAME).write_text(
        json.dumps(_ledger_payload(result, workflow, payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _blocked_result(workflow_id: str, output_dir: Path, blocker: str) -> WorkflowHarnessResult:
    normalized = _normalize_workflow_id(workflow_id)
    ledger_ref = f"{normalized}/{WORKFLOW_HARNESS_LEDGER_NAME}"
    result = WorkflowHarnessResult(
        workflow_id=normalized,
        status="blocked",
        current_state="blocked",
        verification_gate="workflow_known",
        gate_status="blocked",
        evidence_keys=(),
        missing_evidence=(),
        resumable=True,
        ledger_ref=ledger_ref,
        blockers=(blocker,),
        events=(WorkflowHarnessEvent(time.time(), normalized, "blocked", "SlashCommand", blocker, terminal=True),),
    )
    workflow_dir = output_dir / normalized
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / WORKFLOW_HARNESS_LEDGER_NAME).write_text(
        json.dumps(_blocked_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _workflow_definition(workflow_id: str) -> WorkflowDefinition | None:
    normalized = _normalize_workflow_id(workflow_id)
    for workflow in WORKFLOW_DEFINITIONS:
        if workflow.workflow_id == normalized:
            return workflow
    return None


def _normalize_workflow_id(workflow_id: str) -> str:
    normalized = workflow_id.strip().lower().replace("_", "-")
    if WORKFLOW_ID_RE.fullmatch(normalized):
        return normalized
    return "unknown"


def _event(workflow: WorkflowDefinition, state: str, terminal: bool) -> WorkflowHarnessEvent:
    return WorkflowHarnessEvent(
        time.time(),
        workflow.workflow_id,
        state,
        workflow.hook,
        f"{workflow.loop_policy}:{state}",
        terminal=terminal,
    )


def _ledger_payload(result: WorkflowHarnessResult, workflow: WorkflowDefinition, payload: JsonMap) -> JsonMap:
    return {
        "ledger_version": "workflow_harness_v1",
        "workflow_id": result.workflow_id,
        "display_name": workflow.display_name,
        "request_id": _request_id(payload),
        "status": result.status,
        "current_state": result.current_state,
        "verification_gate": result.verification_gate,
        "gate_status": result.gate_status,
        "required_evidence": list(workflow.required_evidence),
        "evidence_keys": list(result.evidence_keys),
        "missing_evidence": list(result.missing_evidence),
        "hook": workflow.hook,
        "loop_policy": workflow.loop_policy,
        "resumable": result.resumable,
        "blockers": list(result.blockers),
        "events": [_event_payload(event) for event in result.events],
    }


def _blocked_payload(result: WorkflowHarnessResult) -> JsonMap:
    return {
        "ledger_version": "workflow_harness_v1",
        "workflow_id": result.workflow_id,
        "status": result.status,
        "current_state": result.current_state,
        "verification_gate": result.verification_gate,
        "gate_status": result.gate_status,
        "required_evidence": [],
        "evidence_keys": list(result.evidence_keys),
        "missing_evidence": list(result.missing_evidence),
        "resumable": result.resumable,
        "blockers": list(result.blockers),
        "events": [_event_payload(event) for event in result.events],
    }


def _event_payload(event: WorkflowHarnessEvent) -> JsonMap:
    return {
        "at": event.at,
        "workflow_id": event.workflow_id,
        "state": event.state,
        "hook": event.hook,
        "summary": event.summary,
        "terminal": event.terminal,
    }


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def _evidence_keys(payload: JsonMap) -> tuple[str, ...]:
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        return tuple(sorted(key for key, value in evidence.items() if isinstance(key, str) and _evidence_value_present(value)))
    evidence_ledger = payload.get("evidence_ledger")
    if isinstance(evidence_ledger, list | tuple):
        keys: set[str] = set()
        for item in evidence_ledger:
            if isinstance(item, str) and item:
                keys.add(item)
            elif isinstance(item, dict):
                key = item.get("key") or item.get("id") or item.get("name")
                if isinstance(key, str) and key:
                    keys.add(key)
        return tuple(sorted(keys))
    return tuple(
        sorted(key for key in payload if isinstance(key, str) and key not in {"request_id", "user_goal", "workflow_id"})
    )


def _evidence_value_present(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | dict | set):
        return bool(value)
    return True
