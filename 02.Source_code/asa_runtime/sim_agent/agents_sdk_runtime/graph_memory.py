from __future__ import annotations

from sim_agent.knowledge import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan
from sim_agent.knowledge.graph_memory import build_graph_brain_context, graph_brain_payload
from sim_agent.schemas._parse import JsonMap


def runtime_graph_memory_payload(payload: JsonMap, role_ids: tuple[str, ...]) -> JsonMap:
    graphdb = payload.get("graphdb")
    database_name = "atomistic_sim_agent_knowledge"
    if isinstance(graphdb, dict) and isinstance(graphdb.get("database_name"), str):
        database_name = graphdb["database_name"]
    gate_plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=(),
            database_name=database_name,
        )
    )
    return graph_brain_payload(build_graph_brain_context(gate_plan, role_ids=role_ids))
