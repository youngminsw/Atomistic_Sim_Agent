from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import WorkflowGate, now, safe_id
from .workflow_harness_payload import text_value


ULTRAGOAL_GOALS_SCHEMA_VERSION: Final = "ultragoal_goals_v1"
ULTRAGOAL_CHECKPOINT_SCHEMA_VERSION: Final = "ultragoal_checkpoint_v1"
EXECUTION_DECISIONS: Final = ("approve", "decline")


@dataclass(frozen=True, slots=True)
class UltragoalArtifactResult:
    refs: tuple[str, ...]


class UltragoalArtifactError(Exception):
    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def materialize_ultragoal_artifacts(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> UltragoalArtifactResult:
    active_goal_id = safe_id(text_value(context.get("goal_id"), "G001"))
    checkpoint_id = safe_id(f"checkpoint-{text_value(context.get('request_id'), active_goal_id)}")
    goals = _goal_items(payload.get("goals"), active_goal_id, text_value(context.get("user_goal"), "ASA workflow goal"))
    brief_body = _brief_body(context, active_goal_id)
    goals_payload = {
        **context,
        "schema_version": ULTRAGOAL_GOALS_SCHEMA_VERSION,
        "artifact_kind": "ultragoal_goals",
        "active_goal_id": active_goal_id,
        "checkpoint_id": checkpoint_id,
        "status": "active",
        "brief_path": "ultragoal/brief.md",
        "ledger_path": "ultragoal/ledger.jsonl",
        "goals": goals,
    }
    checkpoint_payload = {
        **context,
        "schema_version": ULTRAGOAL_CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": "ultragoal_checkpoint",
        "checkpoint_id": checkpoint_id,
        "active_goal_id": active_goal_id,
        "status": "checkpoint_ready",
        "goals_path": "ultragoal/goals.json",
        "signoff_gate_id": "signoff",
        "goal_count": len(goals),
        "checkpoint_hash": _sha256_json(goals_payload),
    }
    _write_once(workflow_dir / "brief.md", brief_body)
    _write_json_once(workflow_dir / "goals.json", goals_payload)
    _write_once(workflow_dir / "ledger.jsonl", json.dumps(checkpoint_payload, sort_keys=True) + "\n")
    return UltragoalArtifactResult(("ultragoal/brief.md", "ultragoal/goals.json", "ultragoal/ledger.jsonl"))


def ultragoal_response_blocker(gate: WorkflowGate, value: object) -> str:
    if gate.workflow_id != "ultragoal" or gate.gate_id not in {"signoff", "execution"}:
        return ""
    decoded = decode_ultragoal_signoff(value)
    return "" if decoded is not None else "ultragoal_signoff_response_mismatch"


def decode_ultragoal_signoff(value: object) -> JsonMap | None:
    if not isinstance(value, dict):
        return None
    allowed_keys = {"decision", "reason"}
    if any(not isinstance(key, str) or key not in allowed_keys for key in value):
        return None
    decision = value.get("decision")
    if not isinstance(decision, str) or decision not in EXECUTION_DECISIONS:
        return None
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        return None
    return {"decision": decision, "approved": decision == "approve", "reason": reason or ""}


def record_ultragoal_signoff_response(output_dir: Path, gate: WorkflowGate, value: object) -> None:
    decoded = decode_ultragoal_signoff(value)
    if decoded is None or gate.workflow_id != "ultragoal" or gate.gate_id not in {"signoff", "execution"}:
        return
    _write_json(
        output_dir / "ultragoal" / "signoff.json",
        {
            "workflow_id": "ultragoal",
            "gate_id": gate.gate_id,
            "answered_at": now(),
            **decoded,
        },
    )


def _goal_items(value: object, active_goal_id: str, default_title: str) -> list[JsonMap]:
    if not isinstance(value, list) or not value:
        return [{"id": active_goal_id, "title": default_title, "status": "in_progress"}]
    goals: list[JsonMap] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        goal_id = safe_id(text_value(item.get("id"), f"G{index:03d}"))
        goals.append(
            {
                "id": goal_id,
                "title": text_value(item.get("title"), text_value(item.get("objective"), f"Goal {index}")),
                "status": text_value(item.get("status"), "in_progress"),
            }
        )
    if goals:
        return goals
    return [{"id": active_goal_id, "title": default_title, "status": "in_progress"}]


def _brief_body(context: JsonMap, active_goal_id: str) -> str:
    return "\n".join(
        (
            "# Ultragoal Brief",
            "",
            f"- request_id: {context['request_id']}",
            f"- active_goal_id: {active_goal_id}",
            f"- objective: {context['user_goal']}",
            "- signoff_gate_id: signoff",
            "",
        )
    )


def _write_once(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise UltragoalArtifactError("ultragoal_artifact_corrupt") from exc
        if current != body:
            raise UltragoalArtifactError("ultragoal_artifact_conflict")
        return
    path.write_text(body, encoding="utf-8")


def _write_json_once(path: Path, payload: JsonMap) -> None:
    _write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_json(payload: JsonMap) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
