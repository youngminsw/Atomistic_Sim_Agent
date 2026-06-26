from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_harness_payload import text_value


JsonEvidenceValue = JsonMap | list[str] | str | int | float | bool | None
VISUAL_QA_SURFACE_SCHEMA_VERSION: Final = "visual_qa_surface_v1"
VISUAL_QA_VERDICT_SCHEMA_VERSION: Final = "visual_qa_verdict_v1"


@dataclass(frozen=True, slots=True)
class VisualQaArtifactResult:
    refs: tuple[str, ...]


class VisualQaArtifactError(Exception):
    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def materialize_visual_qa_artifacts(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> VisualQaArtifactResult:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    surface_ref = text_value(evidence.get("surface_ref"), "")
    screenshot_ref = text_value(evidence.get("screenshot_ref"), "")
    if not surface_ref or not screenshot_ref:
        raise VisualQaArtifactError("visual_qa_surface_required")
    verdict = _oracle_verdict(evidence.get("oracle_verdict"))
    surface_payload = {
        **context,
        "schema_version": VISUAL_QA_SURFACE_SCHEMA_VERSION,
        "artifact_kind": "visual_qa_surface_capture",
        "surface_ref": surface_ref,
        "screenshot_ref": screenshot_ref,
        "capture_target": text_value(evidence.get("capture_target"), "rendered_surface"),
        "stale_artifact_guard": True,
    }
    verdict_payload = {
        **context,
        "schema_version": VISUAL_QA_VERDICT_SCHEMA_VERSION,
        "artifact_kind": "visual_qa_verdict",
        "oracle_verdict": verdict,
        "passed": verdict["passed"],
        "machine_checked": True,
        "surface_capture_path": "visual-qa/surface-capture.json",
        "surface_required_blocker": "visual_qa_surface_required",
    }
    ledger_payload = {
        **context,
        "artifact_kind": "visual_qa_evidence_checkpoint",
        "surface_ref": surface_ref,
        "screenshot_ref": screenshot_ref,
        "passed": verdict["passed"],
        "summary": verdict["summary"],
    }
    _write_json_once(workflow_dir / "surface-capture.json", surface_payload)
    _write_json_once(workflow_dir / "verdict.json", verdict_payload)
    _write_once(workflow_dir / "evidence-ledger.jsonl", json.dumps(ledger_payload, sort_keys=True) + "\n")
    return VisualQaArtifactResult(
        ("visual-qa/surface-capture.json", "visual-qa/verdict.json", "visual-qa/evidence-ledger.jsonl")
    )


def _oracle_verdict(value: JsonEvidenceValue) -> JsonMap:
    loaded = value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VisualQaArtifactError("visual_qa_verdict_invalid") from exc
    if not isinstance(loaded, dict):
        raise VisualQaArtifactError("visual_qa_verdict_invalid")
    passed = loaded.get("passed")
    summary = loaded.get("summary")
    if not isinstance(passed, bool) or not isinstance(summary, str) or not summary.strip():
        raise VisualQaArtifactError("visual_qa_verdict_invalid")
    return {"passed": passed, "summary": summary, "checks": loaded.get("checks", []) if isinstance(loaded.get("checks"), list) else []}


def _write_once(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise VisualQaArtifactError("visual_qa_artifact_corrupt") from exc
        if current != body:
            raise VisualQaArtifactError("visual_qa_artifact_conflict")
        return
    path.write_text(body, encoding="utf-8")


def _write_json_once(path: Path, payload: JsonMap) -> None:
    _write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
