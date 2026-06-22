from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
DESIGN_DOC = PROJECT_ROOT / "docs" / "neo4j-sim-agent-design.md"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_graphdb_gate_dry_run_names_schema_and_layers() -> None:
    from sim_agent.knowledge import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan

    request = GraphDBGateRequest(
        mode=GraphDBMode.DRY_RUN,
        user_db_approval=False,
        existing_database_names=("neo4j", "personal_knowledge"),
    )
    plan = build_graphdb_gate_plan(request)

    assert plan.neo4j_write_enabled is False
    assert plan.database_name == "atomistic_sim_agent_knowledge"
    assert plan.database_role == "empty_demo_database"
    assert plan.requires_empty_database is True
    assert plan.conflict_status == "clear"
    assert "SimAgentSourceItem" in plan.labels
    assert "PhysicsClaim" in plan.labels
    assert "MDRun" in plan.labels
    assert "SimulationArtifact" in plan.labels
    assert "SUPPORTS_CLAIM" in plan.relationships
    assert "PRODUCED_ARTIFACT" in plan.relationships
    assert "PhysicsClaim.record_id IS UNIQUE" in plan.constraints
    assert plan.entity_layers == (
        "literature_facts",
        "md_runs",
        "material_states",
        "surrogate_models",
        "feature_simulations",
        "ui_artifacts",
    )
    assert any("export" in step for step in plan.rollback_steps)


def test_graphdb_import_bundle_writes_replayable_artifacts_and_ingest_report(tmp_path: Path) -> None:
    from sim_agent.knowledge import (
        GraphDBGateRequest,
        GraphDBMode,
        build_graphdb_gate_plan,
        build_source_graph_import_bundle,
        seeded_provenance_registry,
    )

    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=("neo4j", "personal_knowledge"),
        )
    )
    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        plan,
        tmp_path,
        sync_run_id="demo-sync-001",
    )

    report = json.loads(bundle.ingest_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    sources = bundle.sources_path.read_text(encoding="utf-8").splitlines()
    claims = bundle.claims_path.read_text(encoding="utf-8").splitlines()
    retrieval_context = bundle.retrieval_context_path.read_text(encoding="utf-8")

    assert bundle.report.accepted is True
    assert report["accepted"] is True
    assert report["database_role"] == "empty_demo_database"
    assert report["requires_empty_database"] is True
    assert report["neo4j_write_enabled"] is False
    assert report["smoke_query"] == "RETURN 1 AS ok"
    assert len(sources) >= 9
    assert len(claims) >= 9
    assert manifest["artifacts"]["import_cypher"] == "import.cypher"
    cypher = bundle.cypher_path.read_text(encoding="utf-8")
    assert "MERGE (source:SimAgentSourceItem" in cypher
    assert "MERGE (understanding:DocumentUnderstanding" in cypher
    assert "MERGE (entity:CanonicalEntity" in cypher
    assert "HAS_UNDERSTANDING" in cypher
    assert "MENTIONS_ENTITY" in cypher
    assert "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]" in retrieval_context


def test_graphdb_import_bundle_blocks_existing_demo_database_conflict(tmp_path: Path) -> None:
    from sim_agent.knowledge import (
        GraphDBGateRequest,
        GraphDBMode,
        build_graphdb_gate_plan,
        build_source_graph_import_bundle,
        seeded_provenance_registry,
    )

    plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=("atomistic_sim_agent_knowledge",),
        )
    )
    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        plan,
        tmp_path,
        sync_run_id="demo-sync-conflict",
    )

    report = json.loads(bundle.ingest_report_path.read_text(encoding="utf-8"))
    assert bundle.report.accepted is False
    assert report["status"] == "blocked"
    assert report["blocker_reasons"] == ["database_name_conflict"]


def test_graphdb_gate_detects_existing_database_conflict() -> None:
    from sim_agent.knowledge import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan

    request = GraphDBGateRequest(
        mode=GraphDBMode.DRY_RUN,
        user_db_approval=False,
        existing_database_names=("atomistic_sim_agent_knowledge",),
    )
    plan = build_graphdb_gate_plan(request)

    assert plan.neo4j_write_enabled is False
    assert plan.conflict_status == "database_name_conflict"
    assert "atomistic_sim_agent_knowledge" in plan.conflict_checks[0]


def test_graphdb_gate_blocks_write_without_user_approval() -> None:
    from sim_agent.knowledge import GraphDBGateError, GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan

    request = GraphDBGateRequest(
        mode=GraphDBMode.ATTEMPT_WRITE,
        user_db_approval=False,
        existing_database_names=(),
    )

    try:
        build_graphdb_gate_plan(request)
    except GraphDBGateError as exc:
        assert str(exc) == "user_db_approval_required"
    else:
        raise AssertionError("expected GraphDBGateError")


def test_knowledge_db_plan_cli_outputs_dry_run_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "knowledge_db_plan.py"),
            "--dry-run",
            "--existing-db",
            "neo4j",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "neo4j_write_enabled=false" in result.stdout
    assert "database_name=atomistic_sim_agent_knowledge" in result.stdout
    assert "database_role=empty_demo_database" in result.stdout
    assert "requires_empty_database=true" in result.stdout
    assert "labels=" in result.stdout
    assert "relationships=" in result.stdout
    assert "constraints=" in result.stdout


def test_knowledge_db_plan_cli_blocks_write_without_approval() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "knowledge_db_plan.py"),
            "--attempt-write",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "user_db_approval_required" in result.stdout


def test_export_graphdb_import_bundle_cli_writes_empty_demo_bundle(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "export_graphdb_import_bundle.py"),
            "--dry-run",
            "--existing-db",
            "neo4j",
            "--sync-run-id",
            "cli-demo-sync",
            "--out",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((tmp_path / "ingest_report.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "graphdb_ingest_accepted=true" in result.stdout
    assert "database_role=empty_demo_database" in result.stdout
    assert "smoke_query=RETURN 1 AS ok" in result.stdout
    assert report["accepted"] is True
    assert (tmp_path / "sources.jsonl").exists()
    assert (tmp_path / "claims.jsonl").exists()
    assert (tmp_path / "import.cypher").exists()


def test_design_document_records_approval_and_rollback_boundaries() -> None:
    text = DESIGN_DOC.read_text(encoding="utf-8")

    assert "user_db_approval=true" in text
    assert "Rollback" in text
    assert "Conflict Check" in text
    assert "youngminsw/Personal_Knowledge_Agent_Kit" in text
    assert "Literature facts" in text
    assert "MD runs" in text
    assert "Material states" in text
    assert "Surrogate models" in text
    assert "Feature simulations" in text
    assert "UI artifacts" in text
