from __future__ import annotations

from typing import assert_never

from sim_agent.agent_harness import AgentRunResult, OfflineModelClient, RunStatus, SimulationAgentHarness
from sim_agent.agents_sdk_runtime.session_contract import agent_team_session_contract
from sim_agent.agent_harness.types import ToolTraceEvent
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig
from sim_agent.md_campaign import md_campaign_plan_payload
from sim_agent.model_provider_payload import model_provider_payload
from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.schemas.request import RunArtifact


def build_agent_plan_http_response(payload: JsonMap) -> tuple[JsonMap, int]:
    try:
        return _build_agent_plan_http_response(payload)
    except (SchemaValidationError, ModelPolicyError) as exc:
        return {"error": str(exc)}, 400


def _build_agent_plan_http_response(payload: JsonMap) -> tuple[JsonMap, int]:
    endpoint = ModelProviderConfig.from_mapping(model_provider_payload(payload))
    result = SimulationAgentHarness(endpoint=endpoint, client=OfflineModelClient()).plan(payload)
    return _result_payload(result), _status_code(result.status)


def _result_payload(result: AgentRunResult) -> JsonMap:
    clarification = result.clarification
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "final_output": result.final_output,
        "missing_fields": list(clarification.missing_fields) if clarification is not None else [],
        "question": clarification.question if clarification is not None else "",
        "md_campaign_plan": (
            md_campaign_plan_payload(result.md_campaign_plan) if result.md_campaign_plan is not None else None
        ),
        "artifacts": [_artifact_payload(artifact) for artifact in result.artifacts],
        "artifact_count": len(result.artifacts),
        "trace": [_trace_payload(event) for event in result.trace],
        "verification_evidence": list(result.verification_evidence),
        "team_session_contract": agent_team_session_contract(),
    }


def _trace_payload(event: ToolTraceEvent) -> JsonMap:
    return {"tool_name": event.tool_name, "summary": event.summary}


def _artifact_payload(artifact: RunArtifact) -> JsonMap:
    return {
        "artifact_id": artifact.artifact_id,
        "path": artifact.path,
        "artifact_type": artifact.artifact_type,
    }


def _status_code(status: RunStatus) -> int:
    match status:
        case RunStatus.CLARIFICATION_REQUIRED | RunStatus.PLANNED:
            return 200
        case RunStatus.BLOCKED:
            return 409
        case unreachable:
            assert_never(unreachable)
