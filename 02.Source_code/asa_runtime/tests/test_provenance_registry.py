from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_seeded_registry_contains_required_source_backed_physics_claims() -> None:
    from sim_agent.knowledge import seeded_provenance_registry

    registry = seeded_provenance_registry()
    tags = {tag for record in registry.records for tag in record.tags}

    assert len(registry.records) >= 9
    assert {
        "agents_sdk",
        "model_provider",
        "etch_review",
        "force_field",
        "surrogate",
        "level_set",
        "lammps_code_sample",
        "potential_library",
        "qa_rule",
    } <= tags
    assert all(record.source_url for record in registry.records)
    assert registry.list_by_tag("level_set")[0].used_by == ("level_set", "knowledge")


def test_seeded_registry_catalogs_sources_without_storing_source_payloads() -> None:
    from sim_agent.knowledge import seeded_provenance_registry

    registry = seeded_provenance_registry()
    by_id = {record.record_id: record for record in registry.records}

    assert by_id["lammps-ar-si-zbl-template"].source_url.startswith("repo://")
    assert "potential path" in by_id["si-tersoff-potential-library"].claim
    assert "QA gate" in by_id["slurm-job-script-qa-rule"].claim
    assert all("\n" not in record.claim for record in registry.records)


def test_registry_rejects_unsourced_claim() -> None:
    from sim_agent.knowledge import KnowledgeRegistryError, ProvenanceRecord, ProvenanceRegistry, SourceKind

    record = ProvenanceRecord(
        record_id="bad-claim",
        source_url="",
        title="Missing source",
        claim="Ar always etches Si correctly.",
        tags=("etch_review",),
        confidence=0.2,
        extracted_on="2026-06-10",
        used_by=("md",),
        source_kind=SourceKind.PAPER,
    )

    try:
        ProvenanceRegistry(()).with_record(record)
    except KnowledgeRegistryError as exc:
        assert str(exc) == "source_required"
    else:
        raise AssertionError("expected KnowledgeRegistryError")


def test_graphdb_export_is_dry_run_and_names_source_owned_layers() -> None:
    from sim_agent.knowledge import seeded_provenance_registry

    bundle = seeded_provenance_registry().export_graphdb_dry_run()

    assert bundle.neo4j_write_enabled is False
    assert "SimAgentSourceItem" in bundle.labels
    assert "PhysicsClaim" in bundle.labels
    assert "CanonicalEntity" in bundle.labels
    assert "SUPPORTS_CLAIM" in bundle.relationships
    assert "neo4j_write_enabled=false" in bundle.summary_lines()[0]


def test_list_provenance_cli_filters_by_tag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "list_provenance.py"),
            "--tag",
            "level_set",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "source_count=" in result.stdout
    assert "missing_url=false" in result.stdout
    assert "level_set" in result.stdout


def test_add_provenance_cli_rejects_claim_without_source() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "add_provenance.py"),
            "--claim",
            "Ar always etches Si correctly.",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source_required" in result.stdout
