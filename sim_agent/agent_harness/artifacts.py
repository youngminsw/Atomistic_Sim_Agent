from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from sim_agent.md_campaign import md_campaign_plan_payload
from sim_agent.schemas._parse import JsonMap

from .types import AgentRunResult, RunStatus, ToolTraceEvent


@dataclass(frozen=True, slots=True)
class AgentPlanArtifactError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class AgentPlanArtifactBundle:
    output_dir: Path
    manifest_path: Path
    md_campaign_plan_path: Path
    validated_request_path: Path

    @property
    def artifact_count(self) -> int:
        return 3


def write_agent_plan_artifacts(
    output_dir: Path,
    request_payload: JsonMap,
    result: AgentRunResult,
) -> AgentPlanArtifactBundle:
    _require_planned(result)
    if result.md_campaign_plan is None:
        raise AgentPlanArtifactError("md_campaign_plan_required")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    md_campaign_plan_path = output_dir / "md_campaign_plan.json"
    validated_request_path = output_dir / "validated_request.json"
    _write_json(manifest_path, _manifest_payload(result))
    _write_json(md_campaign_plan_path, md_campaign_plan_payload(result.md_campaign_plan))
    _write_json(validated_request_path, request_payload)
    return AgentPlanArtifactBundle(
        output_dir=output_dir,
        manifest_path=manifest_path,
        md_campaign_plan_path=md_campaign_plan_path,
        validated_request_path=validated_request_path,
    )


def _require_planned(result: AgentRunResult) -> None:
    match result.status:
        case RunStatus.PLANNED:
            return
        case RunStatus.CLARIFICATION_REQUIRED | RunStatus.BLOCKED:
            raise AgentPlanArtifactError("planned_result_required")
        case unreachable:
            assert_never(unreachable)


def _manifest_payload(result: AgentRunResult) -> JsonMap:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "artifact_count": len(result.artifacts),
        "artifact_types": [artifact.artifact_type for artifact in result.artifacts],
        "artifacts": {
            artifact.artifact_type: Path(artifact.path).name
            for artifact in result.artifacts
        },
        "trace": [_trace_payload(event) for event in result.trace],
        "verification_evidence": list(result.verification_evidence),
    }


def _trace_payload(event: ToolTraceEvent) -> JsonMap:
    return {"tool_name": event.tool_name, "summary": event.summary}


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
