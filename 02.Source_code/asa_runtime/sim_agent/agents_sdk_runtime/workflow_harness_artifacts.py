from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap
from sim_agent.agents_sdk_runtime.workflow_harness_payload import evidence_keys, request_id, text_value
from sim_agent.agents_sdk_runtime.workflow_harness_types import WorkflowDefinition


@dataclass(frozen=True, slots=True)
class WorkflowArtifactRequest:
    workflow: WorkflowDefinition
    payload: JsonMap
    workflow_dir: Path
    owner_agent_id: str
    target_agent_id: str
    goal_id: str


def materialize_workflow_artifacts(request: WorkflowArtifactRequest) -> tuple[str, ...]:
    request.workflow_dir.mkdir(parents=True, exist_ok=True)
    context: JsonMap = {
        "workflow_id": request.workflow.workflow_id,
        "request_id": request_id(request.payload),
        "goal_id": request.goal_id,
        "owner_agent_id": request.owner_agent_id,
        "target_agent_id": request.target_agent_id,
        "user_goal": text_value(request.payload.get("user_goal"), "Interactive ASA workflow harness"),
        "evidence_keys": list(evidence_keys(request.payload)),
    }
    match request.workflow.workflow_id:
        case "deep-interview":
            return _materialize_deep_interview(request.workflow_dir, context, request.payload)
        case "ralplan":
            return _materialize_ralplan(request.workflow_dir, context, request.workflow)
        case "ultragoal":
            return _materialize_ultragoal(request.workflow_dir, context, request.payload)
        case _:
            path = request.workflow_dir / "workflow-state.json"
            _write_json(path, context | {"artifact_kind": "workflow_state"})
            return (f"{request.workflow.workflow_id}/{path.name}",)


def _materialize_deep_interview(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> tuple[str, ...]:
    transcript = workflow_dir / "transcript.jsonl"
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    question_answer = evidence.get("question_answer", "")
    ambiguity_score = evidence.get("ambiguity_score", "")
    rows = (
        context | {"artifact_kind": "deep_interview_question", "content": text_value(question_answer, "provided")},
        context | {"artifact_kind": "deep_interview_ambiguity_gate", "ambiguity_score": ambiguity_score},
    )
    transcript.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    handoff = workflow_dir / "handoff.md"
    handoff.write_text(
        "\n".join(
            (
                "# Deep Interview Handoff",
                "",
                f"- workflow_id: {context['workflow_id']}",
                f"- goal_id: {context['goal_id']}",
                f"- owner_agent_id: {context['owner_agent_id']}",
                f"- target_agent_id: {context['target_agent_id']}",
                f"- ambiguity_score: {text_value(ambiguity_score, 'provided')}",
                f"- question_answer: {text_value(question_answer, 'provided')}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return ("deep-interview/transcript.jsonl", "deep-interview/handoff.md")


def _materialize_ralplan(workflow_dir: Path, context: JsonMap, workflow: WorkflowDefinition) -> tuple[str, ...]:
    prd = workflow_dir / "prd.md"
    test_spec = workflow_dir / "test-spec.md"
    consensus = workflow_dir / "consensus.json"
    prd.write_text(
        "\n".join(
            (
                "# RALPlan PRD",
                "",
                f"- request_id: {context['request_id']}",
                f"- goal_id: {context['goal_id']}",
                f"- desired_outcome: {context['user_goal']}",
                f"- verification_gate: {workflow.verification_gate}",
                "",
            )
        ),
        encoding="utf-8",
    )
    test_spec.write_text(
        "\n".join(
            (
                "# RALPlan Test Spec",
                "",
                "- happy_path: required workflow artifacts and gate metadata are present",
                "- edge_path: missing PRD/test-spec artifacts block readiness",
                "- regression_path: owner-scoped workflow authority still rejects peer mutation",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        consensus,
        context
        | {
            "artifact_kind": "ralplan_consensus",
            "reviewers": ["planner", "architect", "critic"],
            "decision": "ready_for_execution",
            "prd_path": "ralplan/prd.md",
            "test_spec_path": "ralplan/test-spec.md",
        },
    )
    return ("ralplan/prd.md", "ralplan/test-spec.md", "ralplan/consensus.json")


def _materialize_ultragoal(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> tuple[str, ...]:
    brief = workflow_dir / "brief.md"
    goals = workflow_dir / "goals.json"
    ledger = workflow_dir / "ledger.jsonl"
    goal_items = payload.get("goals")
    if not isinstance(goal_items, list) or not goal_items:
        goal_items = [
            {
                "id": text_value(context.get("goal_id"), "G001"),
                "title": text_value(context.get("user_goal"), "ASA workflow goal"),
                "status": "in_progress",
            }
        ]
    brief.write_text(
        "\n".join(
            (
                "# Ultragoal Brief",
                "",
                f"- request_id: {context['request_id']}",
                f"- active_goal_id: {text_value(context.get('goal_id'), 'G001')}",
                f"- objective: {context['user_goal']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        goals,
        context
        | {
            "artifact_kind": "ultragoal_goals",
            "brief_path": "ultragoal/brief.md",
            "ledger_path": "ultragoal/ledger.jsonl",
            "goals": goal_items,
        },
    )
    ledger.write_text(
        json.dumps(
            context
            | {
                "artifact_kind": "ultragoal_checkpoint",
                "status": "checkpoint_ready",
                "goals_path": "ultragoal/goals.json",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ("ultragoal/brief.md", "ultragoal/goals.json", "ultragoal/ledger.jsonl")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
