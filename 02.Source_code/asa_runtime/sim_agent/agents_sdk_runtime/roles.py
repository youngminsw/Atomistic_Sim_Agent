from __future__ import annotations

from .prompt_assets import load_domain_role_prompt
from .types import AgentRoleDefinition


AGENT_ROLES: tuple[AgentRoleDefinition, ...] = (
    AgentRoleDefinition(
        "md_agent",
        "MD Agent",
        "LAMMPS MD structure/build/run/postprocess/physics gates",
        "handoff_to_md_agent",
        load_domain_role_prompt("md_agent"),
    ),
    AgentRoleDefinition(
        "ml_agent",
        "ML Agent",
        "MD event dataset audit, MDN training, uncertainty, active learning",
        "handoff_to_ml_agent",
        load_domain_role_prompt("ml_agent"),
    ),
    AgentRoleDefinition(
        "feature_scale_agent",
        "Feature Scale Agent",
        "KMC transport and Level-Set profile evolution",
        "handoff_to_feature_scale_agent",
        load_domain_role_prompt("feature_scale_agent"),
    ),
    AgentRoleDefinition(
        "research_agent",
        "Research Agent",
        "Literature search, source-to-graph ingestion, provenance retrieval",
        "handoff_to_research_agent",
        load_domain_role_prompt("research_agent"),
    ),
    AgentRoleDefinition(
        "qa_agent",
        "QA Agent",
        "Run evidence audit, hard blocker checks, final report",
        "handoff_to_qa_agent",
        load_domain_role_prompt("qa_agent"),
    ),
)


def agent_role_ids() -> tuple[str, ...]:
    return tuple(role.role_id for role in AGENT_ROLES)


def agent_role_definition(agent_id: str) -> AgentRoleDefinition | None:
    for role in AGENT_ROLES:
        if role.role_id == agent_id:
            return role
    return None
