from __future__ import annotations

from .types import AgentRoleDefinition


AGENT_ROLES: tuple[AgentRoleDefinition, ...] = (
    AgentRoleDefinition(
        "md_agent",
        "MD Agent",
        "LAMMPS MD structure/build/run/postprocess/physics gates",
        "handoff_to_md_agent",
        "Plan and verify MD work. Never bypass force-field, box-size, physics, or event-quality gates.",
    ),
    AgentRoleDefinition(
        "ml_mdn_agent",
        "ML/MDN Agent",
        "MD event dataset audit, MDN training, uncertainty, active learning",
        "handoff_to_ml_mdn_agent",
        "Train and gate MD-derived surrogate models before feature-scale use.",
    ),
    AgentRoleDefinition(
        "feature_scale_agent",
        "Feature Scale Agent",
        "KMC transport and Level-Set profile evolution",
        "handoff_to_feature_scale_agent",
        "Convert MDN outputs and plasma distributions into profile evolution artifacts.",
    ),
    AgentRoleDefinition(
        "research_graphdb_agent",
        "Research GraphDB Agent",
        "Literature search, source-to-graph ingestion, provenance retrieval",
        "handoff_to_research_graphdb_agent",
        "Build source-backed knowledge with explicit Neo4j write approval boundaries.",
    ),
    AgentRoleDefinition(
        "qa_agent",
        "QA Agent",
        "Run evidence audit, hard blocker checks, final report",
        "handoff_to_qa_agent",
        "Fail runs with missing MD incidents, failed physics gates, or failed GraphDB ingest.",
    ),
)
