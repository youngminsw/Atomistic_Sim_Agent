from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from sim_agent.schemas._parse import JsonMap

from .graphdb_access import AgentGraphContext, GraphDBAccessError, GraphDBConnectionConfig, build_agent_graph_context
from .graphdb_gate import GraphDBGatePlan


@dataclass(frozen=True, slots=True)
class AgentGraphMemorySnapshot:
    agent_id: str
    status: str
    purpose: str
    cypher: str
    parameters: JsonMap
    evidence_count: int
    evidence: tuple[JsonMap, ...]
    blocker: str = ""


@dataclass(frozen=True, slots=True)
class GraphBrainContext:
    status: str
    database_name: str
    smoke_query: str
    research_write_owner: str
    write_requires_approval: bool
    source_owned_labels: tuple[str, ...]
    retrieval_rules: tuple[str, ...]
    agent_snapshots: tuple[AgentGraphMemorySnapshot, ...]


class GraphMemoryReadClient(Protocol):
    def verify_connectivity(self) -> None:
        ...

    def run_read(
        self,
        database_name: str,
        cypher: str,
        parameters: JsonMap,
        limit: int,
    ) -> tuple[JsonMap, ...]:
        ...


class Neo4jGraphMemoryReadClient:
    def __init__(self, config: GraphDBConnectionConfig, *, connection_timeout_s: float = 3.0) -> None:
        username = _first_env_value(config.username_env, "NEO4J_USERNAME", "NEO4J_USER")
        password = _first_env_value(config.password_env, "NEO4J_PASSWORD")
        if username is None:
            raise GraphDBAccessError(f"missing_env:{config.username_env}")
        if password is None:
            raise GraphDBAccessError(f"missing_env:{config.password_env}")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise GraphDBAccessError("neo4j_driver_not_installed") from exc
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(username, password),
            connection_timeout=connection_timeout_s,
            connection_acquisition_timeout=connection_timeout_s,
        )

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def run_read(
        self,
        database_name: str,
        cypher: str,
        parameters: JsonMap,
        limit: int,
    ) -> tuple[JsonMap, ...]:
        from neo4j import READ_ACCESS

        rows: list[JsonMap] = []
        with self._driver.session(database=database_name, default_access_mode=READ_ACCESS) as session:
            result = getattr(session, "run")(cypher, parameters=dict(parameters))
            for record in result:
                rows.append({key: record[key] for key in record.keys()})
                if len(rows) >= limit:
                    break
        return tuple(rows)

    def close(self) -> None:
        self._driver.close()


def build_graph_brain_context(
    gate_plan: GraphDBGatePlan,
    *,
    role_ids: tuple[str, ...],
    connection: GraphDBConnectionConfig | None = None,
    read_client: GraphMemoryReadClient | None = None,
    result_limit: int = 5,
) -> GraphBrainContext:
    context = build_agent_graph_context(gate_plan, connection)
    if read_client is not None:
        read_client.verify_connectivity()
    snapshots = _snapshots(context, role_ids, read_client, result_limit)
    return GraphBrainContext(
        status="read_ready" if read_client is not None else "query_plan_ready",
        database_name=context.database_name,
        smoke_query=context.smoke_query,
        research_write_owner="research_graphdb_agent",
        write_requires_approval=context.write_requires_approval,
        source_owned_labels=context.source_owned_labels,
        retrieval_rules=context.retrieval_rules,
        agent_snapshots=snapshots,
    )


def graph_brain_payload(context: GraphBrainContext) -> JsonMap:
    return {
        "status": context.status,
        "database_name": context.database_name,
        "smoke_query": context.smoke_query,
        "research_write_owner": context.research_write_owner,
        "write_requires_approval": context.write_requires_approval,
        "source_owned_labels": list(context.source_owned_labels),
        "retrieval_rules": list(context.retrieval_rules),
        "agent_snapshots": [
            {
                "agent_id": snapshot.agent_id,
                "status": snapshot.status,
                "purpose": snapshot.purpose,
                "cypher": snapshot.cypher,
                "parameters": snapshot.parameters,
                "evidence_count": snapshot.evidence_count,
                "evidence": list(snapshot.evidence),
                "blocker": snapshot.blocker,
            }
            for snapshot in context.agent_snapshots
        ],
    }


def _snapshots(
    context: AgentGraphContext,
    role_ids: tuple[str, ...],
    read_client: GraphMemoryReadClient | None,
    result_limit: int,
) -> tuple[AgentGraphMemorySnapshot, ...]:
    role_set = set(role_ids)
    return tuple(
        _snapshot(context.database_name, query.agent_id, query.purpose, query.cypher, query.parameters, read_client, result_limit)
        for query in context.role_queries
        if query.agent_id in role_set
    )


def _snapshot(
    database_name: str,
    agent_id: str,
    purpose: str,
    cypher: str,
    parameters: JsonMap,
    read_client: GraphMemoryReadClient | None,
    result_limit: int,
) -> AgentGraphMemorySnapshot:
    if read_client is None:
        return AgentGraphMemorySnapshot(agent_id, "query_planned", purpose, cypher, parameters, 0, ())
    evidence = read_client.run_read(database_name, cypher, parameters, result_limit)
    return AgentGraphMemorySnapshot(agent_id, "read_succeeded", purpose, cypher, parameters, len(evidence), evidence)


def _first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
