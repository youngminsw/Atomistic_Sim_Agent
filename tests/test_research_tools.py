from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_research_tool_imports_source_claim_and_prepares_dry_run_graphdb_bundle() -> None:
    from sim_agent.knowledge import SourceKind, seeded_provenance_registry
    from sim_agent.knowledge.research_tools import ResearchImportRequest, import_research_source

    result = import_research_source(
        seeded_provenance_registry(),
        ResearchImportRequest(
            source_url="https://math.berkeley.edu/~sethian/2006/Papers/sethian.etch3.pdf",
            title="Level set etch source",
            claim="Level-Set profile evolution should be driven by local velocity fields.",
            tags=("level_set", "profile_evolution"),
            used_by=("level_set", "runner"),
            source_kind=SourceKind.PAPER,
            confidence=0.79,
        ),
    )

    assert result.provenance_ready is True
    assert result.graphdb_write is False
    assert result.registry.records[-1].record_id == "level-set-etch-source"
    assert result.graphdb_bundle.neo4j_write_enabled is False
    assert result.graphdb_bundle.records[-1].source_url.endswith("sethian.etch3.pdf")


def test_research_tool_answers_material_question_with_cited_provenance() -> None:
    from sim_agent.knowledge import seeded_provenance_registry
    from sim_agent.knowledge.research_tools import ResearchQuestion, answer_research_question

    answer = answer_research_question(
        seeded_provenance_registry(),
        ResearchQuestion(
            query="Which evidence says force field choice matters for sputtering?",
            tags=("force_field",),
            max_summary_chars=140,
        ),
    )

    assert answer.answer_status == "answered"
    assert len(answer.summary) <= 140
    assert "force-field provenance" in answer.summary
    assert answer.citations[0].record_id == "md-force-field-sensitivity"
    assert answer.citations[0].source_url.startswith("https://")
    assert answer.evidence_record_ids[0] == "md-force-field-sensitivity"
    assert "lammps-ar-si-zbl-template" in answer.evidence_record_ids
    assert "si-tersoff-potential-library" in answer.evidence_record_ids
    assert "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]" in answer.graph_lookup_query


def test_research_tool_rejects_unsourced_claim() -> None:
    from sim_agent.knowledge import seeded_provenance_registry
    from sim_agent.knowledge.research_tools import ResearchImportRequest, ResearchToolError, import_research_source

    request = ResearchImportRequest(
        source_url="",
        title="Unsupported claim",
        claim="Use any force field for all materials.",
        tags=("force_field",),
        used_by=("md",),
    )

    try:
        import_research_source(seeded_provenance_registry(), request)
    except ResearchToolError as exc:
        assert str(exc) == "source_required"
    else:
        raise AssertionError("expected ResearchToolError")


def test_research_import_cli_imports_level_set_source_without_graphdb_write() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "research_import.py"),
            "--url",
            "https://math.berkeley.edu/~sethian/2006/Papers/sethian.etch3.pdf",
            "--tag",
            "level_set",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "provenance_ready=true" in result.stdout
    assert "graphdb_write=false" in result.stdout
    assert "record_id=sethian-etch3" in result.stdout


def test_research_import_cli_rejects_unsourced_claim() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "research_import.py"),
            "--claim",
            "Use any force field for all materials",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "source_required" in result.stdout


def test_research_graphdb_agent_cli_exports_bundle_context_and_query_answer(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "research_graphdb_agent.py"),
            "--out",
            str(tmp_path),
            "--existing-db",
            "neo4j",
            "--database-name",
            "atomistic_sim_agent_demo",
            "--sync-run-id",
            "research-agent-cli-smoke",
            "--query",
            "What LAMMPS and QA knowledge should MD Agent read before a Slurm job?",
            "--tag",
            "lammps_code_sample",
            "--tag",
            "qa_rule",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "research_graphdb_agent_status=ready" in result.stdout
    assert "graphdb_ingest_accepted=true" in result.stdout
    assert "graphdb_write=false" in result.stdout
    assert "agent_access_enabled=true" in result.stdout
    assert "answer_status=answered" in result.stdout
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "agent_graph_context.json").exists()
    assert (tmp_path / "research_answer.json").exists()
    assert (tmp_path / "retrieval_context.md").exists()
