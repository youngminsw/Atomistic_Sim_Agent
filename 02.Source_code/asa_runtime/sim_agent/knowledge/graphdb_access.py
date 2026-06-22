from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .graphdb_gate import GraphDBGatePlan


class GraphDBAccessError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GraphDBConnectionConfig:
    uri: str = "bolt://youngmin-lab:7687"
    database_name: str = "atomistic_sim_agent_knowledge"
    username_env: str = "NEO4J_USERNAME"
    password_env: str = "NEO4J_PASSWORD"


@dataclass(frozen=True, slots=True)
class AgentGraphQuery:
    agent_id: str
    purpose: str
    cypher: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentGraphContext:
    agent_access_enabled: bool
    database_name: str
    database_role: str
    smoke_query: str
    write_requires_approval: bool
    connection: GraphDBConnectionConfig
    role_queries: tuple[AgentGraphQuery, ...]
    source_owned_labels: tuple[str, ...]
    retrieval_rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphDBWriteRequest:
    approve_write: bool
    database_name: str
    require_empty_database: bool = True


@dataclass(frozen=True, slots=True)
class GraphDBWriteReport:
    applied: bool
    status: str
    blocker_reasons: tuple[str, ...]
    database_name: str
    row_counts: dict[str, int]
    executed_statement_kinds: tuple[str, ...]

    def summary_lines(self) -> tuple[str, ...]:
        lines = [
            f"graphdb_write_applied={str(self.applied).lower()}",
            f"graphdb_write_status={self.status}",
            f"database_name={self.database_name}",
        ]
        for key in ("sources", "understandings", "claims", "entities"):
            lines.append(f"row_count_{key}={self.row_counts.get(key, 0)}")
        for blocker in self.blocker_reasons:
            lines.append(f"graphdb_write_blocker={blocker}")
        for kind in self.executed_statement_kinds:
            lines.append(f"executed_statement={kind}")
        return tuple(lines)


class GraphDBClient(Protocol):
    def verify_connectivity(self) -> None:
        ...

    def list_databases(self) -> tuple[str, ...]:
        ...

    def count_nodes(self, database_name: str) -> int:
        ...

    def run_write(self, database_name: str, kind: str, cypher: str, parameters: dict[str, Any]) -> None:
        ...


class Neo4jDriverClient:
    def __init__(self, config: GraphDBConnectionConfig) -> None:
        username = os.environ.get(config.username_env)
        password = os.environ.get(config.password_env)
        if not username:
            raise GraphDBAccessError(f"missing_env:{config.username_env}")
        if not password:
            raise GraphDBAccessError(f"missing_env:{config.password_env}")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise GraphDBAccessError("neo4j_driver_not_installed") from exc
        self._driver = GraphDatabase.driver(config.uri, auth=(username, password))

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def list_databases(self) -> tuple[str, ...]:
        try:
            with self._driver.session(database="system") as session:
                records = session.run("SHOW DATABASES YIELD name RETURN name")
                return tuple(str(record["name"]) for record in records)
        except Exception:
            return ()

    def count_nodes(self, database_name: str) -> int:
        with self._driver.session(database=database_name) as session:
            record = session.run("MATCH (n) RETURN count(n) AS count").single()
            return int(record["count"]) if record is not None else 0

    def run_write(self, database_name: str, kind: str, cypher: str, parameters: dict[str, Any]) -> None:
        with self._driver.session(database=database_name) as session:
            getattr(session, "run")(cypher, **parameters).consume()

    def close(self) -> None:
        self._driver.close()


def build_agent_graph_context(
    gate_plan: GraphDBGatePlan,
    connection: GraphDBConnectionConfig | None = None,
) -> AgentGraphContext:
    resolved_connection = connection or GraphDBConnectionConfig(database_name=gate_plan.database_name)
    return AgentGraphContext(
        agent_access_enabled=True,
        database_name=gate_plan.database_name,
        database_role=gate_plan.database_role,
        smoke_query=gate_plan.smoke_query,
        write_requires_approval=True,
        connection=resolved_connection,
        role_queries=_role_queries(),
        source_owned_labels=gate_plan.source_owned_labels,
        retrieval_rules=(
            "read source-backed claims before selecting force fields or physics assumptions",
            "treat GraphDB as derived evidence, not original source storage",
            "write only through approved staged import bundles",
            "never expose secret values in agent prompts, logs, or graph properties",
        ),
    )


def agent_graph_context_payload(context: AgentGraphContext) -> dict[str, Any]:
    return {
        "agent_access_enabled": context.agent_access_enabled,
        "database_name": context.database_name,
        "database_role": context.database_role,
        "smoke_query": context.smoke_query,
        "write_requires_approval": context.write_requires_approval,
        "connection": {
            "uri": context.connection.uri,
            "database_name": context.connection.database_name,
            "username_env": context.connection.username_env,
            "password_env": context.connection.password_env,
        },
        "role_queries": [
            {
                "agent_id": query.agent_id,
                "purpose": query.purpose,
                "cypher": query.cypher,
                "parameters": query.parameters,
            }
            for query in context.role_queries
        ],
        "source_owned_labels": list(context.source_owned_labels),
        "retrieval_rules": list(context.retrieval_rules),
    }


def execute_graph_import_bundle(
    bundle_dir: Path,
    request: GraphDBWriteRequest,
    *,
    client: GraphDBClient | None = None,
) -> GraphDBWriteReport:
    rows = _load_bundle_rows(bundle_dir)
    blockers = _preflight_blockers(bundle_dir, request, client)
    if blockers:
        return GraphDBWriteReport(
            applied=False,
            status="blocked",
            blocker_reasons=blockers,
            database_name=request.database_name,
            row_counts=_row_counts(rows),
            executed_statement_kinds=(),
        )
    graph_client = client
    created_client = False
    if graph_client is None:
        graph_client = Neo4jDriverClient(GraphDBConnectionConfig(database_name=request.database_name))
        created_client = True
    try:
        graph_client.verify_connectivity()
        executed: list[str] = []
        for kind, cypher, parameters in _write_statements(rows):
            graph_client.run_write(request.database_name, kind, cypher, parameters)
            executed.append(kind)
        return GraphDBWriteReport(
            applied=True,
            status="applied",
            blocker_reasons=(),
            database_name=request.database_name,
            row_counts=_row_counts(rows),
            executed_statement_kinds=tuple(executed),
        )
    finally:
        if created_client and hasattr(graph_client, "close"):
            graph_client.close()  # type: ignore[attr-defined]


def graphdb_write_report_payload(report: GraphDBWriteReport) -> dict[str, Any]:
    return {
        "applied": report.applied,
        "status": report.status,
        "blocker_reasons": list(report.blocker_reasons),
        "database_name": report.database_name,
        "row_counts": report.row_counts,
        "executed_statement_kinds": list(report.executed_statement_kinds),
    }


def _role_queries() -> tuple[AgentGraphQuery, ...]:
    return (
        AgentGraphQuery(
            "orchestrator",
            "retrieve run-level evidence and recent source-backed decisions",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url "
            "ORDER BY claim.confidence DESC LIMIT 10",
            {},
        ),
        AgentGraphQuery(
            "research_graphdb_agent",
            "retrieve literature-backed claims by topic tags",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN $tags) "
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url",
            {"tags": ["force_field", "level_set", "surrogate"]},
        ),
        AgentGraphQuery(
            "md_agent",
            "retrieve force_field and MD validation evidence before LAMMPS planning",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN ['force_field','md','sputtering']) "
            "RETURN claim.record_id, claim.claim, source.source_url ORDER BY claim.confidence DESC",
            {},
        ),
        AgentGraphQuery(
            "ml_mdn_agent",
            "retrieve surrogate, mdn, and uncertainty evidence before model acceptance",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN ['surrogate','mdn','uncertainty']) "
            "RETURN claim.record_id, claim.claim, source.source_url ORDER BY claim.confidence DESC",
            {},
        ),
        AgentGraphQuery(
            "feature_scale_agent",
            "retrieve feature-scale and level-set profile evolution assumptions",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN ['feature_scale','level_set','profile_evolution']) "
            "RETURN claim.record_id, claim.claim, source.source_url ORDER BY claim.confidence DESC",
            {},
        ),
        AgentGraphQuery(
            "qa_agent",
            "retrieve low-confidence and review-needed physics claims for hard-blocker checks",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE claim.needs_review = true OR claim.confidence < 0.75 "
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url",
            {},
        ),
        AgentGraphQuery(
            "infra_agent",
            "retrieve compute, credential, database, and job-script boundary rules before resource actions",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN ['qa_rule','slurm','job_script','endpoint_policy']) "
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url "
            "ORDER BY claim.confidence DESC",
            {},
        ),
    )


def _preflight_blockers(
    bundle_dir: Path,
    request: GraphDBWriteRequest,
    client: GraphDBClient | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not request.approve_write:
        blockers.append("user_db_approval_required")
    if not bundle_dir.exists():
        blockers.append("bundle_dir_not_found")
    report_path = bundle_dir / "ingest_report.json"
    if report_path.exists():
        ingest_report = json.loads(report_path.read_text(encoding="utf-8"))
        if ingest_report.get("accepted") is not True:
            blockers.append("ingest_report_not_accepted")
        if ingest_report.get("database_name") != request.database_name:
            blockers.append("database_name_mismatch")
    else:
        blockers.append("ingest_report_required")
    if blockers:
        return tuple(blockers)
    graph_client = client
    created_client = False
    if graph_client is None:
        graph_client = Neo4jDriverClient(GraphDBConnectionConfig(database_name=request.database_name))
        created_client = True
    try:
        graph_client.verify_connectivity()
        database_names = graph_client.list_databases()
        if database_names and request.database_name not in database_names:
            blockers.append("database_not_found")
        if request.require_empty_database and graph_client.count_nodes(request.database_name) > 0:
            blockers.append("database_not_empty")
    finally:
        if created_client and hasattr(graph_client, "close"):
            graph_client.close()  # type: ignore[attr-defined]
    return tuple(blockers)


def _load_bundle_rows(bundle_dir: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        "sources": _read_jsonl(bundle_dir / "sources.jsonl"),
        "understandings": _read_jsonl(bundle_dir / "understandings.jsonl"),
        "claims": _read_jsonl(bundle_dir / "claims.jsonl"),
        "entities": _read_jsonl(bundle_dir / "canonical_entities.jsonl"),
    }


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        raise GraphDBAccessError(f"bundle_artifact_missing:{path.name}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def _row_counts(rows: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, int]:
    return {key: len(value) for key, value in rows.items()}


def _write_statements(
    rows: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    constraints = (
        (
            "constraint",
            "CREATE CONSTRAINT sim_agent_source_url IF NOT EXISTS "
            "FOR (source:SimAgentSourceItem) REQUIRE source.source_url IS UNIQUE",
            {},
        ),
        (
            "constraint",
            "CREATE CONSTRAINT sim_agent_claim_id IF NOT EXISTS "
            "FOR (claim:PhysicsClaim) REQUIRE claim.record_id IS UNIQUE",
            {},
        ),
        (
            "constraint",
            "CREATE CONSTRAINT sim_agent_understanding_id IF NOT EXISTS "
            "FOR (understanding:DocumentUnderstanding) REQUIRE understanding.understanding_id IS UNIQUE",
            {},
        ),
        (
            "constraint",
            "CREATE CONSTRAINT sim_agent_entity_name IF NOT EXISTS "
            "FOR (entity:CanonicalEntity) REQUIRE entity.name IS UNIQUE",
            {},
        ),
    )
    imports = (
        (
            "sources",
            "UNWIND $sources AS row "
            "MERGE (source:SimAgentSourceItem {source_url: row.source_url}) "
            "SET source += row",
            {"sources": list(rows["sources"])},
        ),
        (
            "understandings",
            "UNWIND $understandings AS row "
            "MATCH (source:SimAgentSourceItem {source_url: row.source_url}) "
            "MERGE (understanding:DocumentUnderstanding {understanding_id: row.understanding_id}) "
            "SET understanding += row "
            "MERGE (source)-[:HAS_UNDERSTANDING {sync_run_id: row.sync_run_id}]->(understanding)",
            {"understandings": list(rows["understandings"])},
        ),
        (
            "claims",
            "UNWIND $claims AS row "
            "MATCH (source:SimAgentSourceItem {source_url: row.source_url}) "
            "MERGE (claim:PhysicsClaim {record_id: row.record_id}) "
            "SET claim += row "
            "MERGE (source)-[:SUPPORTS_CLAIM {sync_run_id: row.sync_run_id}]->(claim)",
            {"claims": list(rows["claims"])},
        ),
        (
            "entities",
            "UNWIND $entities AS row "
            "MERGE (entity:CanonicalEntity {name: row.name}) "
            "SET entity += row "
            "WITH entity, row "
            "UNWIND row.source_record_ids AS record_id "
            "MATCH (claim:PhysicsClaim {record_id: record_id}) "
            "MERGE (claim)-[:MENTIONS_ENTITY {sync_run_id: row.sync_run_id}]->(entity)",
            {"entities": list(rows["entities"])},
        ),
    )
    return constraints + imports
