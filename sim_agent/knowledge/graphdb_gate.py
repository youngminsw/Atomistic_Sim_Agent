from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never


class GraphDBMode(StrEnum):
    DRY_RUN = "dry_run"
    ATTEMPT_WRITE = "attempt_write"


class GraphDBGateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GraphDBGateRequest:
    mode: GraphDBMode
    user_db_approval: bool
    existing_database_names: tuple[str, ...]
    database_name: str = "atomistic_sim_agent_knowledge"
    requires_empty_database: bool = True


@dataclass(frozen=True, slots=True)
class GraphDBGatePlan:
    mode: GraphDBMode
    neo4j_write_enabled: bool
    database_name: str
    database_role: str
    requires_empty_database: bool
    labels: tuple[str, ...]
    relationships: tuple[str, ...]
    constraints: tuple[str, ...]
    entity_layers: tuple[str, ...]
    conflict_status: str
    conflict_checks: tuple[str, ...]
    rollback_steps: tuple[str, ...]
    export_artifacts: tuple[str, ...]
    source_owned_labels: tuple[str, ...]
    smoke_query: str

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"neo4j_write_enabled={str(self.neo4j_write_enabled).lower()}",
            f"database_name={self.database_name}",
            f"database_role={self.database_role}",
            f"requires_empty_database={str(self.requires_empty_database).lower()}",
            f"mode={self.mode.value}",
            f"conflict_status={self.conflict_status}",
            f"labels={','.join(self.labels)}",
            f"relationships={','.join(self.relationships)}",
            f"constraints={';'.join(self.constraints)}",
            f"entity_layers={','.join(self.entity_layers)}",
            f"source_owned_labels={','.join(self.source_owned_labels)}",
            f"smoke_query={self.smoke_query}",
        )


DATABASE_NAME: Final = "atomistic_sim_agent_knowledge"
LABELS: Final = (
    "SimAgentSourceItem",
    "DocumentUnderstanding",
    "PhysicsClaim",
    "CanonicalEntity",
    "ReviewCandidate",
    "SyncRun",
    "MDRun",
    "MaterialState",
    "SurrogateModel",
    "FeatureSimulation",
    "SimulationArtifact",
    "UIArtifact",
)
RELATIONSHIPS: Final = (
    "HAS_UNDERSTANDING",
    "SUPPORTS_CLAIM",
    "MENTIONS_ENTITY",
    "USED_BY_MODULE",
    "NEEDS_REVIEW",
    "USES_MATERIAL_STATE",
    "TRAINED_MODEL",
    "DRIVES_SIMULATION",
    "PRODUCED_ARTIFACT",
    "VISUALIZES_ARTIFACT",
)
CONSTRAINTS: Final = (
    "SimAgentSourceItem.source_url IS UNIQUE",
    "PhysicsClaim.record_id IS UNIQUE",
    "CanonicalEntity.name IS UNIQUE",
    "MDRun.run_id IS UNIQUE",
    "SurrogateModel.kernel_id IS UNIQUE",
    "FeatureSimulation.simulation_id IS UNIQUE",
    "SimulationArtifact.artifact_uri IS UNIQUE",
)
ENTITY_LAYERS: Final = (
    "literature_facts",
    "md_runs",
    "material_states",
    "surrogate_models",
    "feature_simulations",
    "ui_artifacts",
)
ROLLBACK_STEPS: Final = (
    "export JSONL and Cypher plan before any import",
    "tag all staged nodes with sync_run_id before activation",
    "delete only source-owned staged rows for a failed sync_run_id",
    "preserve unrelated labels, databases, and relationships",
)
EXPORT_ARTIFACTS: Final = (
    "sources.jsonl",
    "understandings.jsonl",
    "claims.jsonl",
    "canonical_entities.jsonl",
    "md_runs.jsonl",
    "material_states.jsonl",
    "models.jsonl",
    "simulations.jsonl",
    "ui_artifacts.jsonl",
    "import.cypher",
    "manifest.json",
    "ingest_report.json",
    "retrieval_context.md",
)
SOURCE_OWNED_LABELS: Final = (
    "SimAgentSourceItem",
    "DocumentUnderstanding",
    "PhysicsClaim",
    "MDRun",
    "MaterialState",
    "SurrogateModel",
    "FeatureSimulation",
    "SimulationArtifact",
    "UIArtifact",
)


def build_graphdb_gate_plan(request: GraphDBGateRequest) -> GraphDBGatePlan:
    database_name = _validated_database_name(request.database_name)
    match request.mode:
        case GraphDBMode.DRY_RUN:
            write_enabled = False
        case GraphDBMode.ATTEMPT_WRITE:
            if not request.user_db_approval:
                raise GraphDBGateError("user_db_approval_required")
            write_enabled = False
        case unreachable:
            assert_never(unreachable)

    conflict_status = _conflict_status(database_name, request.existing_database_names, request.requires_empty_database)
    return GraphDBGatePlan(
        mode=request.mode,
        neo4j_write_enabled=write_enabled,
        database_name=database_name,
        database_role="empty_demo_database",
        requires_empty_database=request.requires_empty_database,
        labels=LABELS,
        relationships=RELATIONSHIPS,
        constraints=CONSTRAINTS,
        entity_layers=ENTITY_LAYERS,
        conflict_status=conflict_status,
        conflict_checks=_conflict_checks(database_name, request.existing_database_names, conflict_status),
        rollback_steps=ROLLBACK_STEPS,
        export_artifacts=EXPORT_ARTIFACTS,
        source_owned_labels=SOURCE_OWNED_LABELS,
        smoke_query="RETURN 1 AS ok",
    )


def _validated_database_name(database_name: str) -> str:
    value = database_name.strip()
    if not value:
        raise GraphDBGateError("database_name_required")
    if any(char.isspace() for char in value):
        raise GraphDBGateError("database_name_must_not_contain_whitespace")
    return value


def _conflict_status(
    database_name: str,
    existing_database_names: tuple[str, ...],
    requires_empty_database: bool,
) -> str:
    if requires_empty_database and database_name in existing_database_names:
        return "database_name_conflict"
    return "clear"


def _conflict_checks(database_name: str, existing_database_names: tuple[str, ...], status: str) -> tuple[str, ...]:
    return (
        f"requested_database={database_name}",
        f"existing_databases={','.join(existing_database_names) or '-'}",
        f"status={status}",
    )
