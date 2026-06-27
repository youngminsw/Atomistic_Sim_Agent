from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class ArtifactPathResolution:
    path: Path | None
    blocker: str


def workflow_artifact_validation_blocker(output_dir: Path, workflow_id: str, payload: JsonMap) -> str:
    if workflow_id == "ralplan" and payload.get("validate_artifact_paths") is True:
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return "ralplan_artifact_missing"
        for field in ("prd_path", "test_spec_path"):
            resolved = _artifact_path(output_dir, payload, evidence.get(field))
            if resolved.blocker:
                return resolved.blocker
            if resolved.path is None or not resolved.path.is_file():
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
        resolved = _artifact_path(output_dir, payload, evidence.get(field))
        if resolved.path is None or not resolved.path.is_file():
            missing.append(field)
    return tuple(missing)


def _ultragoal_goals_blocker(output_dir: Path, payload: JsonMap) -> str:
    goals_path = payload.get("goals_path") or payload.get("ultragoal_goals_path")
    if goals_path is None:
        return ""
    resolved = _artifact_path(output_dir, payload, goals_path)
    if resolved.blocker:
        return resolved.blocker
    if resolved.path is None or not resolved.path.is_file():
        return "ultragoal_goals_missing"
    try:
        goals_payload = json.loads(resolved.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "ultragoal_goals_corrupt"
    if not isinstance(goals_payload, dict) or not isinstance(goals_payload.get("goals"), list):
        return "ultragoal_goals_corrupt"
    return ""


def _artifact_path(output_dir: Path, payload: JsonMap, value: str | None) -> ArtifactPathResolution:
    if not isinstance(value, str) or not value:
        return ArtifactPathResolution(None, "")
    path = Path(value)
    if path.is_absolute():
        return ArtifactPathResolution(None, "workflow_artifact_path_untrusted")
    artifact_root = payload.get("artifact_root")
    base = _artifact_root(output_dir, artifact_root)
    root = base.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        return ArtifactPathResolution(None, "workflow_artifact_path_untrusted")
    return ArtifactPathResolution(resolved, "")


def _artifact_root(output_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        return output_dir
    root = Path(value)
    if root.is_absolute():
        return root
    return output_dir / root
