from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


def workflow_artifact_validation_blocker(output_dir: Path, workflow_id: str, payload: JsonMap) -> str:
    if workflow_id == "ralplan" and payload.get("validate_artifact_paths") is True:
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return "ralplan_artifact_missing"
        for field in ("prd_path", "test_spec_path"):
            path = _artifact_path(output_dir, payload, evidence.get(field))
            if path is None or not path.is_file():
                return "ralplan_artifact_missing"
    if workflow_id == "ultragoal":
        return _ultragoal_goals_blocker(output_dir, payload)
    return ""


def workflow_artifact_missing_evidence(output_dir: Path, workflow_id: str, payload: JsonMap) -> tuple[str, ...]:
    if workflow_id != "ralplan" or payload.get("validate_artifact_paths") is not True:
        return ()
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        return ("prd_path", "test_spec_path")
    missing: list[str] = []
    for field in ("prd_path", "test_spec_path"):
        path = _artifact_path(output_dir, payload, evidence.get(field))
        if path is None or not path.is_file():
            missing.append(field)
    return tuple(missing)


def _ultragoal_goals_blocker(output_dir: Path, payload: JsonMap) -> str:
    goals_path = payload.get("goals_path") or payload.get("ultragoal_goals_path")
    if goals_path is None:
        return ""
    path = _artifact_path(output_dir, payload, goals_path)
    if path is None or not path.is_file():
        return "ultragoal_goals_missing"
    try:
        goals_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "ultragoal_goals_corrupt"
    if not isinstance(goals_payload, dict) or not isinstance(goals_payload.get("goals"), list):
        return "ultragoal_goals_corrupt"
    return ""


def _artifact_path(output_dir: Path, payload: JsonMap, value: str | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    artifact_root = payload.get("artifact_root")
    base = Path(artifact_root) if isinstance(artifact_root, str) and artifact_root else output_dir
    return base / path
