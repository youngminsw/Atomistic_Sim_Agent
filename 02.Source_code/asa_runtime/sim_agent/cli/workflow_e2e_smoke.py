from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke
from sim_agent.agents_sdk_runtime.workflow_runtime import respond_workflow_gate
from sim_agent.agents_sdk_runtime.workflow_harness_types import WorkflowHarnessResult
from sim_agent.schemas._parse import JsonMap


WORKFLOW_COMMANDS: Final = ("/deep-interview", "/ralplan", "/ultragoal", "/visual-qa", "/ultraresearch")
SKILL_IDS: Final = ("insane-search",)
SUBAGENT_WORKFLOW_DENIAL: Final = "persistent_workflow_surface_unavailable_for_bounded_subagent"
SUBAGENT_SKILL_DENIAL: Final = "persistent_skill_surface_unavailable_for_bounded_subagent"


@dataclass(frozen=True, slots=True)
class WorkflowE2ESmokeRequest:
    output_dir: Path
    scenario: str


@dataclass(frozen=True, slots=True)
class WorkflowE2ESmokeResult:
    status: str
    output_json: Path
    transcript_path: Path
    blockers: tuple[str, ...]


def run_workflow_e2e_smoke(request: WorkflowE2ESmokeRequest) -> WorkflowE2ESmokeResult:
    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(request.scenario)
    results = tuple(_run_workflow(command.removeprefix("/"), output_dir, run_id) for command in WORKFLOW_COMMANDS)
    artifact_refs = _artifact_refs(results)
    payload = _e2e_payload(request.scenario, output_dir, run_id, results, artifact_refs)
    transcript = _transcript(request.scenario, run_id, payload)
    output_json = output_dir / "workflow-e2e.json"
    transcript_path = output_dir / "workflow-transcript.txt"
    _write_json(output_json, payload)
    transcript_path.write_text(transcript, encoding="utf-8")
    blockers = tuple(str(blocker) for blocker in payload["blockers"])
    return WorkflowE2ESmokeResult(str(payload["status"]), output_json, transcript_path, blockers)


def _run_workflow(workflow_id: str, output_dir: Path, run_id: str) -> WorkflowHarnessResult:
    payload = _workflow_payload(workflow_id, run_id)
    if workflow_id == "visual-qa":
        _write_visual_capture(output_dir, run_id)
    result = run_workflow_harness_smoke(workflow_id, payload, output_dir)
    if workflow_id == "deep-interview" and result.gate is not None:
        respond_workflow_gate(
            output_dir,
            {
                "workflow_id": workflow_id,
                "gate_id": result.gate["gate_id"],
                "responder_agent_id": "orchestrator",
                "idempotency_key": f"{run_id}-deep-interview",
                "value": {"selected": ["ready"], "handoff_ready": True},
            },
        )
        return run_workflow_harness_smoke(workflow_id, payload, output_dir)
    if workflow_id in {"ralplan", "ultragoal"} and result.gate is not None:
        respond_workflow_gate(
            output_dir,
            {
                "workflow_id": workflow_id,
                "gate_id": result.gate["gate_id"],
                "responder_agent_id": "orchestrator",
                "idempotency_key": f"{run_id}-{workflow_id}",
                "value": _approval_value(),
            },
        )
        return run_workflow_harness_smoke(workflow_id, payload, output_dir)
    return result


def _workflow_payload(workflow_id: str, run_id: str) -> JsonMap:
    base: JsonMap = {
        "request_id": run_id,
        "goal_id": f"goal-{run_id}",
        "user_goal": "Close ASA workflow parity through runtime-backed e2e evidence.",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "actor_agent_id": "orchestrator",
    }
    match workflow_id:  # noqa: MATCH_OK - workflow IDs are plugin strings with a generic fallback.
        case "deep-interview":
            return base | {
                "deep_interview": {
                    "round": 1,
                    "round_id": f"round-{run_id}",
                    "question_id": "q1",
                    "multi": False,
                    "options": ["ready"],
                    "ambiguity": 0.1,
                }
            }
        case "ralplan":
            return base | {
                "evidence": {"prd_path": "plans/prd.md", "test_spec_path": "plans/test-spec.md"},
                "gate": _approval_gate(),
            }
        case "ultragoal":
            goal: JsonMap = {"id": f"G-{run_id}", "title": "workflow parity", "status": "in_progress"}
            return base | {
                "evidence": {"codex_goal_snapshot": "active"},
                "goals": [goal],
                "gate": _signoff_gate(),
            }
        case "visual-qa":
            verdict: JsonMap = {
                "passed": True,
                "summary": "workflow rows visible",
                "checks": ["all-workflows"],
            }
            evidence: JsonMap = {
                "surface_ref": "tui://workflow-panel",
                "screenshot_ref": "captures/workflow-panel.txt",
                "oracle_verdict": verdict,
            }
            return base | {"evidence": evidence}
        case "ultraresearch":
            return base | {"evidence": _ultraresearch_evidence(run_id)}
        case _:
            return base


def _approval_gate() -> JsonMap:
    return {
        "gate_id": "approval",
        "gate_kind": "response_schema",
        "response_schema": {
            "type": "object",
            "required": ["decision"],
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "request-changes", "reject"]},
                "comments": {"type": "string"},
            },
        },
    }


def _signoff_gate() -> JsonMap:
    return {
        "gate_id": "signoff",
        "gate_kind": "response_schema",
        "response_schema": {
            "type": "object",
            "required": ["decision"],
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "decline"]},
                "reason": {"type": "string"},
            },
        },
    }


def _approval_value() -> JsonMap:
    return {"decision": "approve"}


def _write_visual_capture(output_dir: Path, run_id: str) -> None:
    capture = output_dir / "captures" / "workflow-panel.txt"
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_text(f"workflow panel capture for {run_id}\n", encoding="utf-8")


def _ultraresearch_evidence(run_id: str) -> JsonMap:
    return {
        "research_question": "What public evidence supports ASA workflow parity?",
        "source_journal": f"journals/{run_id}.jsonl",
        "insane_search_trace": {
            "skill_id": "insane_search",
            "surface": "skill",
            "ok": True,
            "public_only": True,
            "ssrf_safe": True,
            "auth_required": False,
            "grid_exhausted": False,
            "untried_routes": [],
            "must_invoke_playwright_mcp": False,
            "stop_reason": "success",
            "routes": ["phase0", "fetch_chain"],
            "sources": [
                {
                    "url": "https://example.com/public/asa-workflows",
                    "route": "fetch_chain",
                    "title": "ASA workflow public evidence",
                    "evidence_ref": "trace[0]",
                }
            ],
            "trace": [
                {
                    "phase": "probe",
                    "executor": "curl_cffi",
                    "url": "https://example.com/public/asa-workflows",
                    "status": 200,
                    "verdict": "weak_ok",
                }
            ],
        },
    }


def _artifact_refs(results: tuple[WorkflowHarnessResult, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for result in results:
        refs.append(result.ledger_ref)
        refs.extend(result.artifact_refs)
    return tuple(dict.fromkeys(refs))


def _e2e_payload(
    scenario: str,
    output_dir: Path,
    run_id: str,
    results: tuple[WorkflowHarnessResult, ...],
    artifact_refs: tuple[str, ...],
) -> JsonMap:
    blockers = _result_blockers(results)
    return {
        "schema_version": "asa_workflow_e2e_smoke_v1",
        "scenario": scenario,
        "run_id": run_id,
        "status": "succeeded" if not blockers else "blocked",
        "blockers": list(blockers),
        "workflow_ids": list(WORKFLOW_COMMANDS),
        "skill_ids": list(SKILL_IDS),
        "workflow_results": [_result_row(result) for result in results],
        "ledger_paths": [result.ledger_ref for result in results],
        "artifacts": list(artifact_refs),
        "artifact_hashes": {ref: _sha256(output_dir / ref) for ref in artifact_refs},
        "bounded_subagent_denials": _bounded_subagent_denials(),
    }


def _result_row(result: WorkflowHarnessResult) -> JsonMap:
    return {
        "workflow_id": f"/{result.workflow_id}",
        "status": result.status,
        "current_state": result.current_state,
        "gate_status": result.gate_status,
        "ledger_ref": result.ledger_ref,
        "artifact_refs": list(result.artifact_refs),
        "blockers": list(result.blockers),
    }


def _bounded_subagent_denials() -> JsonMap:
    denials: dict[str, str] = {command: SUBAGENT_WORKFLOW_DENIAL for command in WORKFLOW_COMMANDS}
    denials["insane-search"] = SUBAGENT_SKILL_DENIAL
    return denials


def _result_blockers(results: tuple[WorkflowHarnessResult, ...]) -> tuple[str, ...]:
    blockers: list[str] = []
    for result in results:
        if result.status != "ready":
            blockers.append(f"{result.workflow_id}:{result.status}")
        blockers.extend(result.blockers)
    return tuple(dict.fromkeys(blockers))


def _transcript(scenario: str, run_id: str, payload: JsonMap) -> str:
    lines = [f"workflow_e2e_smoke_status={payload['status']}", f"scenario={scenario}", f"run_id={run_id}"]
    lines.extend(str(command) for command in payload["workflow_ids"])
    lines.append("skill=insane-search")
    lines.extend(f"artifact={ref}" for ref in payload["artifacts"])
    return "\n".join(lines) + "\n"


def _run_id(scenario: str) -> str:
    safe_scenario = scenario.replace("_", "-").replace("/", "-")
    return f"{safe_scenario}-{int(time.time() * 1000)}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
