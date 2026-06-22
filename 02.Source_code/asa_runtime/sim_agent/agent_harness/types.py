from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sim_agent.md_campaign import MDCampaignPlan
from sim_agent.schemas.request import RunArtifact


class RunStatus(StrEnum):
    CLARIFICATION_REQUIRED = "clarification_required"
    PLANNED = "planned"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ClarificationRequired:
    missing_fields: tuple[str, ...]
    question: str


@dataclass(frozen=True, slots=True)
class ToolTraceEvent:
    tool_name: str
    summary: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    final_output: str
    clarification: ClarificationRequired | None
    md_campaign_plan: MDCampaignPlan | None
    artifacts: tuple[RunArtifact, ...]
    trace: tuple[ToolTraceEvent, ...]
    verification_evidence: tuple[str, ...]
