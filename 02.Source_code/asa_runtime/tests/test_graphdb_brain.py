from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
import tomllib


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str


def test_project_declares_neo4j_driver_and_default_brain_config() -> None:
    from sim_agent.runtime_config import default_runtime_config

    pyproject = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    config = default_runtime_config()

    assert any(dependency.startswith("neo4j") for dependency in pyproject["project"]["dependencies"])
    assert config.graphdb.uri == "bolt://youngmin-lab:7687"
    assert config.graphdb.uri_env == "NEO4J_URI"
    assert config.graphdb.user_env == "NEO4J_USERNAME"
    assert config.graphdb.password_env == "NEO4J_PASSWORD"
    assert config.graphdb.database == "atomistic_sim_agent_knowledge"


def test_graph_brain_context_plans_read_queries_for_every_runtime_agent() -> None:
    from sim_agent.knowledge import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan
    from sim_agent.knowledge.graph_memory import build_graph_brain_context, graph_brain_payload

    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=("neo4j",),
        )
    )
    brain = build_graph_brain_context(
        plan,
        role_ids=(
            "orchestrator",
            "md_agent",
            "ml_mdn_agent",
            "feature_scale_agent",
            "research_graphdb_agent",
            "qa_agent",
        ),
    )
    payload = graph_brain_payload(brain)

    assert payload["status"] == "query_plan_ready"
    assert payload["research_write_owner"] == "research_graphdb_agent"
    assert payload["write_requires_approval"] is True
    snapshots = tuple(as_mapping(item, "agent_snapshot") for item in as_sequence(payload["agent_snapshots"], "agent_snapshots"))
    assert {as_str(item["agent_id"], "agent_id") for item in snapshots} == {
        "orchestrator",
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "research_graphdb_agent",
        "qa_agent",
    }
    assert any(
        as_str(item["agent_id"], "agent_id") == "md_agent" and "force_field" in as_str(item["purpose"], "purpose")
        for item in snapshots
    )
    assert all(item["status"] == "query_planned" for item in snapshots)


def test_graph_brain_context_reads_evidence_with_injected_client() -> None:
    from sim_agent.knowledge import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan
    from sim_agent.knowledge.graph_memory import build_graph_brain_context, graph_brain_payload

    fake_client = FakeGraphMemoryReadClient(
        rows=(
            {
                "record_id": "ff-001",
                "claim": "Use source-backed force-field evidence before MD.",
                "confidence": 0.9,
                "source_url": "https://example.test/paper",
            },
        )
    )
    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=("neo4j",),
        )
    )

    brain = build_graph_brain_context(plan, role_ids=("md_agent",), read_client=fake_client)
    payload = graph_brain_payload(brain)

    assert fake_client.verified is True
    assert fake_client.queries[0]["database_name"] == "atomistic_sim_agent_knowledge"
    assert payload["status"] == "read_ready"
    snapshot = as_mapping(as_sequence(payload["agent_snapshots"], "agent_snapshots")[0], "agent_snapshot")
    evidence = as_mapping(as_sequence(snapshot["evidence"], "evidence")[0], "evidence")
    assert snapshot["status"] == "read_succeeded"
    assert snapshot["evidence_count"] == 1
    assert evidence["record_id"] == "ff-001"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeGraphMemoryReadClient:
    """Mutable fake records read calls and connectivity state during one test."""

    rows: tuple[JsonMap, ...]
    verified: bool = False
    queries: list[JsonMap] = field(default_factory=list)

    def verify_connectivity(self) -> None:
        self.verified = True

    def run_read(
        self,
        database_name: str,
        cypher: str,
        parameters: JsonMap,
        limit: int,
    ) -> tuple[JsonMap, ...]:
        self.queries.append(
            {
                "database_name": database_name,
                "cypher": cypher,
                "parameters": parameters,
                "limit": limit,
            }
        )
        return self.rows[:limit]
