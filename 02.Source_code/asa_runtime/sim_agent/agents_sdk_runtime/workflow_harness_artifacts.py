from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap
from sim_agent.agents_sdk_runtime.workflow_harness_payload import evidence_keys, request_id, text_value
from sim_agent.agents_sdk_runtime.workflow_ralplan import RalplanArtifactError, materialize_ralplan_artifacts
from sim_agent.agents_sdk_runtime.workflow_ultragoal import (
    UltragoalArtifactError,
    materialize_ultragoal_artifacts,
)
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
        case "visual-qa":
            return _materialize_visual_qa(request.workflow_dir, context, request.payload)
        case "ultraresearch":
            return _materialize_ultraresearch(request.workflow_dir, context, request.payload)
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
    result = materialize_ralplan_artifacts(workflow_dir, context, workflow.verification_gate)
    prd = workflow_dir / "prd.md"
    test_spec = workflow_dir / "test-spec.md"
    _write_once(
        prd,
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
    )
    _write_once(
        test_spec,
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
    )
    return result.refs


def _write_once(path: Path, body: str) -> None:
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RalplanArtifactError("ralplan_artifact_corrupt") from exc
        if current != body:
            raise RalplanArtifactError("ralplan_artifact_conflict")
        return
    path.write_text(body, encoding="utf-8")


def _materialize_ultragoal(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> tuple[str, ...]:
    return materialize_ultragoal_artifacts(workflow_dir, context, payload).refs


def _materialize_visual_qa(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> tuple[str, ...]:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    surface_capture = workflow_dir / "surface-capture.json"
    verdict = workflow_dir / "verdict.json"
    _write_json(
        surface_capture,
        context
        | {
            "artifact_kind": "visual_qa_surface_capture",
            "surface_ref": evidence.get("surface_ref", ""),
            "screenshot_ref": evidence.get("screenshot_ref", ""),
            "capture_target": evidence.get("capture_target", "rendered_surface"),
            "stale_artifact_guard": True,
        },
    )
    _write_json(
        verdict,
        context
        | {
            "artifact_kind": "visual_qa_verdict",
            "oracle_verdict": evidence.get("oracle_verdict", {}),
            "surface_capture_path": "visual-qa/surface-capture.json",
            "surface_required_blocker": "visual_qa_surface_required",
        },
    )
    return ("visual-qa/surface-capture.json", "visual-qa/verdict.json")


def _materialize_ultraresearch(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> tuple[str, ...]:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    acquisition_plan = workflow_dir / "acquisition-plan.json"
    journal = workflow_dir / "research-journal.jsonl"
    research_question = text_value(evidence.get("research_question"), context["user_goal"])
    insane_search_trace = evidence.get("insane_search_trace") if isinstance(evidence.get("insane_search_trace"), dict) else {}
    _write_json(
        acquisition_plan,
        context
        | {
            "artifact_kind": "ultraresearch_acquisition_plan",
            "research_question": research_question,
            "source_journal": evidence.get("source_journal", "ultraresearch/research-journal.jsonl"),
            "insane_search": {
                "surface": "skill",
                "skill_id": "insane_search",
                "public_only": True,
                "ssrf_safe": True,
                "auth_required": False,
                "trace": insane_search_trace,
            },
            "content_policy": {
                "untrusted_web_content_is_evidence_only": True,
                "credentialed_or_paywalled_sources_denied": True,
            },
        },
    )
    rows = (
        context
        | {
            "artifact_kind": "ultraresearch_question_decomposition",
            "research_question": research_question,
            "axes": ["source_discovery", "claim_verification", "synthesis"],
        },
        context
        | {
            "artifact_kind": "ultraresearch_acquisition_wave",
            "route": "insane_search",
            "trace": insane_search_trace,
            "source_journal": evidence.get("source_journal", "ultraresearch/research-journal.jsonl"),
        },
        context
        | {
            "artifact_kind": "ultraresearch_synthesis_checkpoint",
            "status": "synthesis_ready",
            "citation_required": True,
        },
    )
    journal.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return ("ultraresearch/acquisition-plan.json", "ultraresearch/research-journal.jsonl")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
