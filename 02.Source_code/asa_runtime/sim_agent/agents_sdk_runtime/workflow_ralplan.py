from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import WorkflowGate, now, safe_id


RALPLAN_STAGE_SCHEMA_VERSION: Final = "ralplan_stage_v1"
APPROVAL_DECISIONS: Final = ("approve", "request-changes", "reject")


@dataclass(frozen=True, slots=True)
class RalplanArtifactResult:
    refs: tuple[str, ...]


class RalplanArtifactError(Exception):
    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def materialize_ralplan_artifacts(workflow_dir: Path, context: JsonMap, verification_gate: str) -> RalplanArtifactResult:
    run_id = safe_id(str(context.get("request_id") or "ralplan-run"))
    plan_dir = workflow_dir / "plans" / run_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    stages = (
        ("planner", 1, _planner_markdown(context, verification_gate)),
        ("architect", 2, _review_markdown(context, "architect", "APPROVE")),
        ("critic", 3, _review_markdown(context, "critic", "OKAY")),
        ("final", 4, _final_markdown(context)),
    )
    stage_refs: list[dict[str, object]] = []
    for stage, stage_n, body in stages:
        path = plan_dir / f"stage-{stage_n:02d}-{stage}.md"
        _write_once(path, body)
        stage_refs.append(
            {
                "schema_version": RALPLAN_STAGE_SCHEMA_VERSION,
                "stage": stage,
                "stage_n": stage_n,
                "path": _rel(workflow_dir, path),
                "sha256": _sha256_text(body),
            }
        )
    pending = plan_dir / "pending-approval.md"
    _write_once(pending, stages[-1][2])
    _write_index(plan_dir / "index.jsonl", stage_refs)
    consensus = {
        **context,
        "artifact_kind": "ralplan_consensus",
        "run_id": run_id,
        "current_phase": "pending_approval",
        "approved": False,
        "reviewers": ["planner", "architect", "critic"],
        "decision": "pending_approval",
        "prd_path": "ralplan/prd.md",
        "test_spec_path": "ralplan/test-spec.md",
        "pending_approval_path": f"ralplan/{_rel(workflow_dir, pending)}",
        "stage_artifacts": stage_refs,
    }
    _write_json_once(workflow_dir / "consensus.json", consensus)
    return RalplanArtifactResult(
        (
            "ralplan/prd.md",
            "ralplan/test-spec.md",
            "ralplan/consensus.json",
            f"ralplan/{_rel(workflow_dir, pending)}",
            f"ralplan/{_rel(workflow_dir, plan_dir / 'index.jsonl')}",
        )
    )


def ralplan_consensus_payload(workflow_dir: Path, context: JsonMap, verification_gate: str) -> JsonMap:
    materialize_ralplan_artifacts(workflow_dir, context, verification_gate)
    consensus_path = workflow_dir / "consensus.json"
    loaded = json.loads(consensus_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RalplanArtifactError("ralplan_consensus_corrupt")
    return loaded


def write_ralplan_consensus(workflow_dir: Path, consensus: JsonMap) -> None:
    _write_json_once(workflow_dir / "consensus.json", consensus)


def ralplan_response_blocker(gate: WorkflowGate, value: object) -> str:
    if gate.workflow_id != "ralplan" or gate.gate_id != "approval":
        return ""
    decoded = decode_ralplan_approval(value)
    return "" if decoded is not None else "ralplan_approval_response_mismatch"


def decode_ralplan_approval(value: object) -> JsonMap | None:
    if not isinstance(value, dict):
        return None
    allowed_keys = {"decision", "comments"}
    if any(not isinstance(key, str) or key not in allowed_keys for key in value):
        return None
    decision = value.get("decision")
    if not isinstance(decision, str) or decision not in APPROVAL_DECISIONS:
        return None
    comments = value.get("comments")
    if comments is not None and not isinstance(comments, str):
        return None
    if decision == "request-changes" and (comments is None or not comments.strip()):
        return None
    return {"decision": decision, "approved": decision == "approve", "comments": comments or ""}


def record_ralplan_approval_response(output_dir: Path, gate: WorkflowGate, value: object) -> None:
    decoded = decode_ralplan_approval(value)
    if decoded is None or gate.workflow_id != "ralplan" or gate.gate_id != "approval":
        return
    workflow_dir = output_dir / "ralplan"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        workflow_dir / "approval.json",
        {
            "workflow_id": "ralplan",
            "gate_id": gate.gate_id,
            "answered_at": now(),
            **decoded,
        },
    )


def _write_index(path: Path, rows: list[JsonMap]) -> None:
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _write_once(path, body)


def _write_once(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RalplanArtifactError("ralplan_artifact_corrupt") from exc
        if current != body:
            raise RalplanArtifactError("ralplan_artifact_conflict")
        return
    path.write_text(body, encoding="utf-8")


def _write_json_once(path: Path, payload: JsonMap) -> None:
    _write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _planner_markdown(context: JsonMap, verification_gate: str) -> str:
    return "\n".join(
        (
            "# RALPlan PRD",
            "",
            f"- request_id: {context['request_id']}",
            f"- goal_id: {context['goal_id']}",
            f"- desired_outcome: {context['user_goal']}",
            f"- verification_gate: {verification_gate}",
            "",
        )
    )


def _review_markdown(context: JsonMap, stage: str, verdict: str) -> str:
    return "\n".join(
        (
            f"# RALPlan {stage.title()} Review",
            "",
            f"- request_id: {context['request_id']}",
            f"- verdict: {verdict}",
            "- notes: artifact persisted through the ASA workflow runtime",
            "",
        )
    )


def _final_markdown(context: JsonMap) -> str:
    return "\n".join(
        (
            "# RALPlan Pending Approval",
            "",
            "## Decision",
            f"Plan {context['goal_id']} is ready for explicit approval.",
            "",
            "## Follow-ups",
            "- Approve, request changes, or reject through the ralplan approval gate.",
            "",
        )
    )


def _sha256_text(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
