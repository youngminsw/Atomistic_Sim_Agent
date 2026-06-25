from __future__ import annotations

import re
import time
from dataclasses import dataclass
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
    actor_agent_id: str = "orchestrator"
    owner_agent_id: str = "orchestrator"
    target_agent_id: str = "orchestrator"
    goal_id: str = ""
    gate: JsonMap | None = None
    artifact_refs: tuple[str, ...] = ()


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


def workflow_definition(workflow_id: str) -> WorkflowDefinition | None:
    normalized = normalize_workflow_id(workflow_id)
    for workflow in WORKFLOW_DEFINITIONS:
        if workflow.workflow_id == normalized:
            return workflow
    return None


def normalize_workflow_id(workflow_id: str) -> str:
    normalized = workflow_id.strip().lower().replace("_", "-")
    if WORKFLOW_ID_RE.fullmatch(normalized):
        return normalized
    return "unknown"


def workflow_event(workflow: WorkflowDefinition, state: str, terminal: bool) -> WorkflowHarnessEvent:
    return WorkflowHarnessEvent(
        time.time(),
        workflow.workflow_id,
        state,
        workflow.hook,
        f"{workflow.loop_policy}:{state}",
        terminal=terminal,
    )
