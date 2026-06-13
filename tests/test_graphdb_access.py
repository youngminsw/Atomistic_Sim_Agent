from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_agent_graph_context_exposes_queries_for_every_agent_role() -> None:
    from sim_agent.knowledge import (
        GraphDBConnectionConfig,
        GraphDBGateRequest,
        GraphDBMode,
        agent_graph_context_payload,
        build_agent_graph_context,
        build_graphdb_gate_plan,
    )

    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=("neo4j",),
            database_name="atomistic_sim_agent_demo",
        )
    )
    context = build_agent_graph_context(
        plan,
        GraphDBConnectionConfig(
            uri="bolt://youngmin-lab:7687",
            database_name="atomistic_sim_agent_demo",
            username_env="NEO4J_USER",
            password_env="NEO4J_PASSWORD",
        ),
    )
    payload = agent_graph_context_payload(context)

    assert payload["agent_access_enabled"] is True
    assert payload["database_name"] == "atomistic_sim_agent_demo"
    assert payload["smoke_query"] == "RETURN 1 AS ok"
    assert payload["write_requires_approval"] is True
    assert payload["connection"]["uri"] == "bolt://youngmin-lab:7687"
    assert payload["connection"]["username_env"] == "NEO4J_USER"
    assert payload["connection"]["password_env"] == "NEO4J_PASSWORD"
    assert "password" not in payload["connection"]
    assert {item["agent_id"] for item in payload["role_queries"]} == {
        "orchestrator",
        "research_graphdb_agent",
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "qa_agent",
        "infra_agent",
    }
    assert all("MATCH" in item["cypher"] for item in payload["role_queries"])
    assert any("force_field" in item["purpose"] for item in payload["role_queries"])


def test_graphdb_import_executor_blocks_write_without_approval(tmp_path: Path) -> None:
    from sim_agent.knowledge import (
        GraphDBGateRequest,
        GraphDBMode,
        GraphDBWriteRequest,
        build_graphdb_gate_plan,
        build_source_graph_import_bundle,
        execute_graph_import_bundle,
        seeded_provenance_registry,
    )

    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        build_graphdb_gate_plan(
            GraphDBGateRequest(
                mode=GraphDBMode.DRY_RUN,
                user_db_approval=False,
                existing_database_names=("neo4j",),
            )
        ),
        tmp_path / "bundle",
        sync_run_id="product-graphdb-blocked",
    )
    fake_client = FakeGraphDBClient()

    report = execute_graph_import_bundle(
        bundle.output_dir,
        GraphDBWriteRequest(
            approve_write=False,
            database_name=bundle.report.database_name,
            require_empty_database=True,
        ),
        client=fake_client,
    )

    assert report.applied is False
    assert report.status == "blocked"
    assert report.blocker_reasons == ("user_db_approval_required",)
    assert fake_client.executed == []


def test_graphdb_import_executor_applies_bundle_with_injected_client(tmp_path: Path) -> None:
    from sim_agent.knowledge import (
        GraphDBGateRequest,
        GraphDBMode,
        GraphDBWriteRequest,
        build_graphdb_gate_plan,
        build_source_graph_import_bundle,
        execute_graph_import_bundle,
        graphdb_write_report_payload,
        seeded_provenance_registry,
    )

    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        build_graphdb_gate_plan(
            GraphDBGateRequest(
                mode=GraphDBMode.DRY_RUN,
                user_db_approval=False,
                existing_database_names=("neo4j",),
                database_name="atomistic_sim_agent_demo",
            )
        ),
        tmp_path / "bundle",
        sync_run_id="product-graphdb-apply",
    )
    fake_client = FakeGraphDBClient(existing_database_names=("atomistic_sim_agent_demo",), node_count=0)

    report = execute_graph_import_bundle(
        bundle.output_dir,
        GraphDBWriteRequest(
            approve_write=True,
            database_name="atomistic_sim_agent_demo",
            require_empty_database=True,
        ),
        client=fake_client,
    )
    payload = graphdb_write_report_payload(report)

    assert report.applied is True
    assert report.status == "applied"
    assert report.blocker_reasons == ()
    assert fake_client.verified is True
    assert [item["kind"] for item in fake_client.executed] == [
        "constraint",
        "constraint",
        "constraint",
        "constraint",
        "sources",
        "understandings",
        "claims",
        "entities",
    ]
    assert payload["row_counts"]["sources"] >= 9
    assert payload["row_counts"]["claims"] >= 9
    assert payload["row_counts"]["entities"] >= 28


def test_graphdb_import_executor_closes_owned_clients(tmp_path: Path, monkeypatch: Any) -> None:
    from sim_agent.knowledge import (
        GraphDBGateRequest,
        GraphDBMode,
        GraphDBWriteRequest,
        build_graphdb_gate_plan,
        build_source_graph_import_bundle,
        seeded_provenance_registry,
    )
    import sim_agent.knowledge.graphdb_access as graphdb_access

    created_clients: list[OwnedFakeGraphDBClient] = []

    class OwnedFakeGraphDBClient(FakeGraphDBClient):
        def __init__(self, config: Any) -> None:
            super().__init__(existing_database_names=("atomistic_sim_agent_demo",), node_count=0)
            self.config = config
            self.closed = False
            created_clients.append(self)

        def close(self) -> None:
            self.closed = True

    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        build_graphdb_gate_plan(
            GraphDBGateRequest(
                mode=GraphDBMode.DRY_RUN,
                user_db_approval=False,
                existing_database_names=("neo4j",),
                database_name="atomistic_sim_agent_demo",
            )
        ),
        tmp_path / "bundle",
        sync_run_id="product-graphdb-owned-client-close",
    )
    monkeypatch.setattr(graphdb_access, "Neo4jDriverClient", OwnedFakeGraphDBClient)

    report = graphdb_access.execute_graph_import_bundle(
        bundle.output_dir,
        GraphDBWriteRequest(
            approve_write=True,
            database_name="atomistic_sim_agent_demo",
            require_empty_database=True,
        ),
    )

    assert report.applied is True
    assert len(created_clients) == 2
    assert all(client.closed for client in created_clients)


def test_apply_graphdb_import_bundle_cli_blocks_without_approval(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    export = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "export_graphdb_import_bundle.py"),
            "--dry-run",
            "--existing-db",
            "neo4j",
            "--sync-run-id",
            "cli-product-graphdb",
            "--out",
            str(bundle_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert export.returncode == 0, export.stdout + export.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "apply_graphdb_import_bundle.py"),
            "--bundle-dir",
            str(bundle_dir),
            "--database-name",
            "atomistic_sim_agent_knowledge",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "graphdb_write_status=blocked" in result.stdout
    assert "graphdb_write_blocker=user_db_approval_required" in result.stdout


@dataclass
class FakeGraphDBClient:
    existing_database_names: tuple[str, ...] = ()
    node_count: int = 0
    verified: bool = False
    executed: list[dict[str, Any]] = field(default_factory=list)

    def verify_connectivity(self) -> None:
        self.verified = True

    def list_databases(self) -> tuple[str, ...]:
        return self.existing_database_names

    def count_nodes(self, database_name: str) -> int:
        return self.node_count

    def run_write(self, database_name: str, kind: str, cypher: str, parameters: dict[str, Any]) -> None:
        self.executed.append(
            {
                "database_name": database_name,
                "kind": kind,
                "cypher": cypher,
                "parameters": parameters,
            }
        )
