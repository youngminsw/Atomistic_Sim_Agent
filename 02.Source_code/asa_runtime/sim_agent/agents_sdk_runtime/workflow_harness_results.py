from __future__ import annotations

import time
from pathlib import Path

from sim_agent.agents_sdk_runtime.workflow_harness_ledger import write_blocked_workflow_ledger
from sim_agent.agents_sdk_runtime.workflow_harness_types import (
    WORKFLOW_HARNESS_LEDGER_NAME,
    WorkflowDefinition,
    WorkflowHarnessEvent,
    WorkflowHarnessResult,
    normalize_workflow_id,
)


def missing_evidence_blocker(workflow: WorkflowDefinition) -> str:
    match workflow.workflow_id:  # noqa: MATCH_OK - workflow IDs are plugin strings with a generic fallback.
        case "visual-qa":
            return "visual_qa_surface_required"
        case "ultraresearch":
            return "ultraresearch_evidence_required"
        case _:
            return "workflow_gate_missing_evidence"


def blocked_result(workflow_id: str, output_dir: Path, blocker: str) -> WorkflowHarnessResult:
    normalized = normalize_workflow_id(workflow_id)
    result = WorkflowHarnessResult(
        workflow_id=normalized,
        status="blocked",
        current_state="blocked",
        verification_gate="workflow_known",
        gate_status="blocked",
        evidence_keys=(),
        missing_evidence=(),
        resumable=True,
        ledger_ref=f"{normalized}/{WORKFLOW_HARNESS_LEDGER_NAME}",
        blockers=(blocker,),
        events=(WorkflowHarnessEvent(time.time(), normalized, "blocked", "SlashCommand", blocker, terminal=True),),
    )
    write_blocked_workflow_ledger(output_dir / normalized, result)
    return result
