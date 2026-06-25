from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap
from sim_agent.agents_sdk_runtime.workflow_harness_payload import request_id
from sim_agent.agents_sdk_runtime.workflow_harness_types import (
    WORKFLOW_HARNESS_LEDGER_NAME,
    WorkflowDefinition,
    WorkflowHarnessEvent,
    WorkflowHarnessResult,
)


def write_workflow_ledger(workflow_dir: Path, result: WorkflowHarnessResult, workflow: WorkflowDefinition, payload: JsonMap) -> None:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / WORKFLOW_HARNESS_LEDGER_NAME).write_text(
        json.dumps(_ledger_payload(result, workflow, payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_blocked_workflow_ledger(workflow_dir: Path, result: WorkflowHarnessResult) -> None:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / WORKFLOW_HARNESS_LEDGER_NAME).write_text(
        json.dumps(_blocked_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ledger_payload(result: WorkflowHarnessResult, workflow: WorkflowDefinition, payload: JsonMap) -> JsonMap:
    ledger: JsonMap = {
        "ledger_version": "workflow_harness_v1",
        "workflow_id": result.workflow_id,
        "display_name": workflow.display_name,
        "request_id": request_id(payload),
        "actor_agent_id": result.actor_agent_id,
        "owner_agent_id": result.owner_agent_id,
        "target_agent_id": result.target_agent_id,
        "goal_id": result.goal_id,
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
        "artifact_refs": list(result.artifact_refs),
        "events": [_event_payload(event) for event in result.events],
    }
    if result.gate is not None:
        ledger = dict(ledger) | {"gate": result.gate}
    return ledger


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
