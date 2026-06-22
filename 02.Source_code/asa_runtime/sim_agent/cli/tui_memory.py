from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.knowledge.memory_seed import MemorySeedError, build_memory_seed_bundle, read_memory_seed_sources_from_neo4j
from sim_agent.knowledge import (
    GraphDBAccessError,
    GraphDBConnectionConfig,
    GraphDBGateRequest,
    GraphDBMode,
    build_graphdb_gate_plan,
)
from sim_agent.knowledge.graph_memory import (
    GraphBrainContext,
    Neo4jGraphMemoryReadClient,
    build_graph_brain_context,
    graph_brain_payload,
)
from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str


ROLE_IDS: tuple[str, ...] = (
    "orchestrator",
    "md_agent",
    "ml_mdn_agent",
    "feature_scale_agent",
    "research_graphdb_agent",
    "qa_agent",
)


def handle_memory(args: Sequence[str], output_stream: TextIO) -> None:
    output_stream.write("GraphDB Brain\n")
    output_stream.write("graph_memory=true\n")
    connection = _connection_from_runtime_config()
    if args and args[0] == "seed":
        _write_memory_seed(output_stream)
        return
    if args and args[0] == "live":
        _write_live_memory(connection, output_stream)
        return
    brain = _brain(connection)
    _write_memory_payload(graph_brain_payload(brain), output_stream)
    output_stream.write("live_check_hint=/memory live\n")


def _write_live_memory(connection: GraphDBConnectionConfig, output_stream: TextIO) -> None:
    client: Neo4jGraphMemoryReadClient | None = None
    try:
        client = Neo4jGraphMemoryReadClient(connection)
        brain = _brain(connection, client)
    except GraphDBAccessError as exc:
        output_stream.write("graph_memory_status=blocked\n")
        output_stream.write(f"graph_memory_blocker={exc}\n")
        return
    except _neo4j_error_types() as exc:
        output_stream.write("graph_memory_status=blocked\n")
        output_stream.write(f"graph_memory_blocker={exc.__class__.__name__}\n")
        return
    finally:
        if client is not None:
            client.close()
    _write_memory_payload(graph_brain_payload(brain), output_stream)


def _brain(
    connection: GraphDBConnectionConfig,
    read_client: Neo4jGraphMemoryReadClient | None = None,
) -> GraphBrainContext:
    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=(),
            database_name=connection.database_name,
        )
    )
    return build_graph_brain_context(plan, role_ids=ROLE_IDS, connection=connection, read_client=read_client)


def _write_memory_payload(payload: JsonMap, output_stream: TextIO) -> None:
    output_stream.write(f"graph_memory_status={as_str(payload['status'], 'status')}\n")
    output_stream.write(f"database_name={as_str(payload['database_name'], 'database_name')}\n")
    output_stream.write(f"research_write_owner={as_str(payload['research_write_owner'], 'research_write_owner')}\n")
    output_stream.write(f"write_requires_approval={str(payload['write_requires_approval']).lower()}\n")
    for item in as_sequence(payload["agent_snapshots"], "agent_snapshots"):
        snapshot = as_mapping(item, "agent_snapshot")
        output_stream.write(
            f"agent_brain={as_str(snapshot['agent_id'], 'agent_id')}:"
            f"{as_str(snapshot['status'], 'status')}:"
            f"evidence={snapshot['evidence_count']}\n"
        )


def _write_memory_seed(output_stream: TextIO) -> None:
    config = load_runtime_config()
    output_root = config.evidence_root
    try:
        sources = read_memory_seed_sources_from_neo4j()
        bundle = build_memory_seed_bundle(
            Path(output_root) / "graphdb-memory-seed",
            database_name=config.graphdb.database,
            sync_run_id="tui-personal-memory-seed",
            memory_sources=sources,
        )
    except MemorySeedError as exc:
        output_stream.write("memory_seed_status=blocked\n")
        output_stream.write(f"memory_seed_blocker={exc}\n")
        return
    output_stream.write("memory_seed_status=ready\n")
    output_stream.write(f"memory_seed_source_count={len(sources)}\n")
    output_stream.write(f"memory_seed_bundle_dir={bundle.output_dir}\n")
    output_stream.write(f"memory_seed_ingest_report={bundle.ingest_report_path}\n")
    output_stream.write(f"memory_seed_config_root={output_root}\n")
    output_stream.write("memory_seed_write=false\n")


def _connection_from_runtime_config() -> GraphDBConnectionConfig:
    graphdb = load_runtime_config().graphdb
    return GraphDBConnectionConfig(
        uri=os.environ.get(graphdb.uri_env, graphdb.uri),
        database_name=graphdb.database,
        username_env=graphdb.user_env,
        password_env=graphdb.password_env,
    )


def _neo4j_error_types() -> tuple[type[BaseException], ...]:
    try:
        from neo4j.exceptions import Neo4jError, ServiceUnavailable
    except ImportError:
        return (RuntimeError,)
    return (Neo4jError, ServiceUnavailable)
