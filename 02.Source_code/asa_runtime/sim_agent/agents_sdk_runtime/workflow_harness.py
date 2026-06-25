from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap
from sim_agent.agents_sdk_runtime.workflow_harness_artifacts import (
    WorkflowArtifactRequest,
    materialize_workflow_artifacts,
)
from sim_agent.agents_sdk_runtime.workflow_harness_ledger import (
    write_blocked_workflow_ledger,
    write_workflow_ledger,
)
from sim_agent.agents_sdk_runtime.workflow_harness_payload import agent_id, evidence_keys, optional_gate
from sim_agent.agents_sdk_runtime.workflow_harness_types import (
    WORKFLOW_HARNESS_LEDGER_NAME,
    WorkflowDefinition,
    WorkflowHarnessEvent,
    WorkflowHarnessResult,
    normalize_workflow_id,
    workflow_definition,
    workflow_event,
    workflow_harness_catalog,
)
from sim_agent.agents_sdk_runtime.workflow_runtime import (
    WorkflowRuntimeStartRequest,
    start_workflow_runtime,
    workflow_authority_blocker,
)


@dataclass(frozen=True, slots=True)
class HarnessContext:
    actor_agent_id: str
    owner_agent_id: str
    target_agent_id: str
    goal_id: str
    ledger_ref: str
    workflow_dir: Path


def run_workflow_harness_smoke(workflow_id: str, payload: JsonMap, output_dir: Path) -> WorkflowHarnessResult:
    workflow = workflow_definition(workflow_id)
    if workflow is None:
        return _blocked_result(workflow_id, output_dir, "unknown_workflow")
    context = _harness_context(workflow, payload, output_dir)
    authority_blocker = workflow_authority_blocker(
        context.actor_agent_id,
        context.owner_agent_id,
        context.target_agent_id,
    )
    if authority_blocker:
        events = (
            workflow_event(workflow, "initialized", False),
            WorkflowHarnessEvent(
                time.time(),
                workflow.workflow_id,
                "blocked",
                workflow.hook,
                f"{workflow.verification_gate}:authority_blocked",
                terminal=True,
            ),
        )
        result = WorkflowHarnessResult(
            workflow_id=workflow.workflow_id,
            status="blocked",
            current_state="blocked",
            verification_gate=workflow.verification_gate,
            gate_status="blocked",
            evidence_keys=evidence_keys(payload),
            missing_evidence=(),
            resumable=True,
            ledger_ref=context.ledger_ref,
            blockers=(authority_blocker,),
            events=events,
            actor_agent_id=context.actor_agent_id,
            owner_agent_id=context.owner_agent_id,
            target_agent_id=context.target_agent_id,
            goal_id=context.goal_id,
        )
        write_workflow_ledger(context.workflow_dir, result, workflow, payload)
        return result
    present_evidence = evidence_keys(payload)
    missing = tuple(key for key in workflow.required_evidence if key not in present_evidence)
    if missing:
        events = (
            workflow_event(workflow, "initialized", False),
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
            evidence_keys=present_evidence,
            missing_evidence=missing,
            resumable=True,
            ledger_ref=context.ledger_ref,
            blockers=("workflow_gate_missing_evidence",),
            events=events,
            actor_agent_id=context.actor_agent_id,
            owner_agent_id=context.owner_agent_id,
            target_agent_id=context.target_agent_id,
            goal_id=context.goal_id,
        )
        write_workflow_ledger(context.workflow_dir, result, workflow, payload)
        return result
    artifact_refs = materialize_workflow_artifacts(
        WorkflowArtifactRequest(
            workflow,
            payload,
            context.workflow_dir,
            context.owner_agent_id,
            context.target_agent_id,
            context.goal_id,
        )
    )
    runtime = start_workflow_runtime(
        WorkflowRuntimeStartRequest(
            output_dir,
            workflow.workflow_id,
            context.actor_agent_id,
            context.owner_agent_id,
            context.target_agent_id,
            context.goal_id,
            payload,
            optional_gate(payload),
        )
    )
    if runtime.status == "blocked":
        events = (
            workflow_event(workflow, "initialized", False),
            WorkflowHarnessEvent(
                time.time(),
                workflow.workflow_id,
                "blocked",
                workflow.hook,
                f"{workflow.verification_gate}:{runtime.gate_status}",
                terminal=True,
            ),
        )
        result = WorkflowHarnessResult(
            workflow_id=workflow.workflow_id,
            status="blocked",
            current_state="blocked",
            verification_gate=workflow.verification_gate,
            gate_status=runtime.gate_status,
            evidence_keys=present_evidence,
            missing_evidence=runtime.missing_evidence,
            resumable=True,
            ledger_ref=context.ledger_ref,
            blockers=runtime.blockers,
            events=events,
            actor_agent_id=context.actor_agent_id,
            owner_agent_id=context.owner_agent_id,
            target_agent_id=context.target_agent_id,
            goal_id=context.goal_id,
            gate=runtime.gate.to_json() if runtime.gate is not None else None,
            artifact_refs=artifact_refs,
        )
        write_workflow_ledger(context.workflow_dir, result, workflow, payload)
        return result
    events = tuple(workflow_event(workflow, state, state == workflow.current_state) for state in workflow.states)
    result = WorkflowHarnessResult(
        workflow_id=workflow.workflow_id,
        status="ready",
        current_state=workflow.current_state,
        verification_gate=workflow.verification_gate,
        gate_status="passed",
        evidence_keys=present_evidence,
        missing_evidence=(),
        resumable=True,
        ledger_ref=context.ledger_ref,
        blockers=(),
        events=events,
        actor_agent_id=context.actor_agent_id,
        owner_agent_id=context.owner_agent_id,
        target_agent_id=context.target_agent_id,
        goal_id=context.goal_id,
        gate=runtime.gate.to_json() if runtime.gate is not None else None,
        artifact_refs=artifact_refs,
    )
    write_workflow_ledger(context.workflow_dir, result, workflow, payload)
    return result


def _harness_context(workflow: WorkflowDefinition, payload: JsonMap, output_dir: Path) -> HarnessContext:
    owner_agent_id = agent_id(payload, "owner_agent_id", "orchestrator")
    return HarnessContext(
        actor_agent_id=agent_id(payload, "actor_agent_id", agent_id(payload, "caller_agent_id", owner_agent_id)),
        owner_agent_id=owner_agent_id,
        target_agent_id=agent_id(payload, "target_agent_id", owner_agent_id),
        goal_id=agent_id(payload, "goal_id", ""),
        ledger_ref=f"{workflow.workflow_id}/{WORKFLOW_HARNESS_LEDGER_NAME}",
        workflow_dir=output_dir / workflow.workflow_id,
    )


def _blocked_result(workflow_id: str, output_dir: Path, blocker: str) -> WorkflowHarnessResult:
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
