from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graphdb_gate import GraphDBGatePlan
from .provenance_registry import ProvenanceRegistry
from .types import ProvenanceRecord


class GraphImportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GraphIngestReport:
    accepted: bool
    status: str
    blocker_reasons: tuple[str, ...]
    sync_run_id: str
    database_name: str
    database_role: str
    requires_empty_database: bool
    neo4j_write_enabled: bool
    smoke_query: str
    source_count: int
    claim_count: int
    entity_count: int
    bounded_summary_max_chars: int
    artifact_paths: tuple[str, ...]

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"graphdb_ingest_accepted={str(self.accepted).lower()}",
            f"graphdb_ingest_status={self.status}",
            f"database_name={self.database_name}",
            f"database_role={self.database_role}",
            f"requires_empty_database={str(self.requires_empty_database).lower()}",
            f"neo4j_write_enabled={str(self.neo4j_write_enabled).lower()}",
            f"smoke_query={self.smoke_query}",
            f"source_count={self.source_count}",
            f"claim_count={self.claim_count}",
            f"entity_count={self.entity_count}",
        )


@dataclass(frozen=True, slots=True)
class GraphImportBundle:
    output_dir: Path
    manifest_path: Path
    ingest_report_path: Path
    sources_path: Path
    understandings_path: Path
    claims_path: Path
    entities_path: Path
    cypher_path: Path
    retrieval_context_path: Path
    report: GraphIngestReport

    def summary_lines(self) -> tuple[str, ...]:
        return self.report.summary_lines() + (
            f"manifest_path={self.manifest_path}",
            f"ingest_report_path={self.ingest_report_path}",
            f"sources_path={self.sources_path}",
            f"claims_path={self.claims_path}",
            f"import_cypher_path={self.cypher_path}",
            f"retrieval_context_path={self.retrieval_context_path}",
        )


def build_source_graph_import_bundle(
    registry: ProvenanceRegistry,
    gate_plan: GraphDBGatePlan,
    output_dir: Path,
    *,
    sync_run_id: str,
    bounded_summary_max_chars: int = 420,
) -> GraphImportBundle:
    sync_run_id = _validated_sync_run_id(sync_run_id)
    if bounded_summary_max_chars < 120:
        raise GraphImportError("bounded_summary_max_chars_too_small")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _source_rows(registry.records, sync_run_id)
    understanding_rows = _understanding_rows(registry.records, sync_run_id, bounded_summary_max_chars)
    claim_rows = _claim_rows(registry.records, sync_run_id, bounded_summary_max_chars)
    entity_rows = _entity_rows(registry.records, sync_run_id)
    blockers = _blocker_reasons(registry.records, gate_plan)
    status = "ready_for_approved_import" if not blockers else "blocked"

    sources_path = output_dir / "sources.jsonl"
    understandings_path = output_dir / "understandings.jsonl"
    claims_path = output_dir / "claims.jsonl"
    entities_path = output_dir / "canonical_entities.jsonl"
    cypher_path = output_dir / "import.cypher"
    retrieval_context_path = output_dir / "retrieval_context.md"
    manifest_path = output_dir / "manifest.json"
    ingest_report_path = output_dir / "ingest_report.json"

    _write_jsonl(sources_path, source_rows)
    _write_jsonl(understandings_path, understanding_rows)
    _write_jsonl(claims_path, claim_rows)
    _write_jsonl(entities_path, entity_rows)
    cypher_path.write_text(_import_cypher(gate_plan), encoding="utf-8")
    retrieval_context_path.write_text(_retrieval_context(gate_plan), encoding="utf-8")

    artifact_paths = (
        str(sources_path),
        str(understandings_path),
        str(claims_path),
        str(entities_path),
        str(cypher_path),
        str(retrieval_context_path),
        str(manifest_path),
        str(ingest_report_path),
    )
    report = GraphIngestReport(
        accepted=not blockers,
        status=status,
        blocker_reasons=blockers,
        sync_run_id=sync_run_id,
        database_name=gate_plan.database_name,
        database_role=gate_plan.database_role,
        requires_empty_database=gate_plan.requires_empty_database,
        neo4j_write_enabled=gate_plan.neo4j_write_enabled,
        smoke_query=gate_plan.smoke_query,
        source_count=len(source_rows),
        claim_count=len(claim_rows),
        entity_count=len(entity_rows),
        bounded_summary_max_chars=bounded_summary_max_chars,
        artifact_paths=artifact_paths,
    )
    ingest_report_path.write_text(_json(report_payload(report)) + "\n", encoding="utf-8")
    manifest_path.write_text(_json(_manifest_payload(gate_plan, report)) + "\n", encoding="utf-8")
    return GraphImportBundle(
        output_dir=output_dir,
        manifest_path=manifest_path,
        ingest_report_path=ingest_report_path,
        sources_path=sources_path,
        understandings_path=understandings_path,
        claims_path=claims_path,
        entities_path=entities_path,
        cypher_path=cypher_path,
        retrieval_context_path=retrieval_context_path,
        report=report,
    )


def report_payload(report: GraphIngestReport) -> dict[str, Any]:
    return {
        "accepted": report.accepted,
        "status": report.status,
        "blocker_reasons": list(report.blocker_reasons),
        "sync_run_id": report.sync_run_id,
        "database_name": report.database_name,
        "database_role": report.database_role,
        "requires_empty_database": report.requires_empty_database,
        "neo4j_write_enabled": report.neo4j_write_enabled,
        "smoke_query": report.smoke_query,
        "source_count": report.source_count,
        "claim_count": report.claim_count,
        "entity_count": report.entity_count,
        "bounded_summary_max_chars": report.bounded_summary_max_chars,
        "artifact_paths": list(report.artifact_paths),
    }


def _blocker_reasons(records: tuple[ProvenanceRecord, ...], gate_plan: GraphDBGatePlan) -> tuple[str, ...]:
    blockers: list[str] = []
    if not records:
        blockers.append("source_records_required")
    if gate_plan.conflict_status != "clear":
        blockers.append(gate_plan.conflict_status)
    duplicate_ids = _duplicates(tuple(record.record_id for record in records))
    if duplicate_ids:
        blockers.append(f"duplicate_record_ids:{','.join(duplicate_ids)}")
    for record in records:
        if not record.source_url:
            blockers.append(f"source_url_required:{record.record_id or '-'}")
        if not record.claim:
            blockers.append(f"claim_required:{record.record_id or '-'}")
        if not record.tags:
            blockers.append(f"tags_required:{record.record_id or '-'}")
        if not record.used_by:
            blockers.append(f"used_by_required:{record.record_id or '-'}")
        if not 0.0 <= record.confidence <= 1.0:
            blockers.append(f"confidence_out_of_range:{record.record_id or '-'}")
    return tuple(blockers)


def _source_rows(records: tuple[ProvenanceRecord, ...], sync_run_id: str) -> tuple[dict[str, Any], ...]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.source_url in rows:
            rows[record.source_url]["claim_record_ids"].append(record.record_id)
            rows[record.source_url]["tags"] = sorted(set(rows[record.source_url]["tags"]) | set(record.tags))
            continue
        rows[record.source_url] = {
            "source_id": _stable_id("source", record.source_url),
            "source_url": record.source_url,
            "title": record.title,
            "source_kind": record.source_kind.value,
            "status": "present",
            "sync_run_id": sync_run_id,
            "claim_record_ids": [record.record_id],
            "tags": list(record.tags),
            "source_owned_label": "SimAgentSourceItem",
        }
    return tuple(rows[key] for key in sorted(rows))


def _understanding_rows(
    records: tuple[ProvenanceRecord, ...],
    sync_run_id: str,
    max_chars: int,
) -> tuple[dict[str, Any], ...]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = _stable_id("source", record.source_url)
        if source_id in rows:
            rows[source_id]["evidence_record_ids"].append(record.record_id)
            continue
        rows[source_id] = {
            "understanding_id": _stable_id("understanding", record.source_url),
            "source_id": source_id,
            "source_url": record.source_url,
            "bounded_summary": _bounded_text(record.claim, max_chars),
            "purpose": _purpose(record),
            "important_terms": list(record.tags),
            "source_evidence": record.source_url,
            "extraction_status": "bounded_summary_ready",
            "evidence_record_ids": [record.record_id],
            "sync_run_id": sync_run_id,
            "source_owned_label": "DocumentUnderstanding",
        }
    return tuple(rows[key] for key in sorted(rows))


def _claim_rows(
    records: tuple[ProvenanceRecord, ...],
    sync_run_id: str,
    max_chars: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "record_id": record.record_id,
            "source_id": _stable_id("source", record.source_url),
            "source_url": record.source_url,
            "claim": record.claim,
            "bounded_summary": _bounded_text(record.claim, max_chars),
            "tags": list(record.tags),
            "confidence": record.confidence,
            "used_by": list(record.used_by),
            "extracted_on": record.extracted_on,
            "needs_review": record.confidence < 0.75,
            "sync_run_id": sync_run_id,
            "source_owned_label": "PhysicsClaim",
        }
        for record in sorted(records, key=lambda item: item.record_id)
    )


def _entity_rows(records: tuple[ProvenanceRecord, ...], sync_run_id: str) -> tuple[dict[str, Any], ...]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        for tag in record.tags:
            key = f"topic:{tag}"
            rows.setdefault(
                key,
                {
                    "entity_id": _stable_id("entity", key),
                    "name": tag,
                    "entity_type": "topic",
                    "source_record_ids": [],
                    "sync_run_id": sync_run_id,
                },
            )["source_record_ids"].append(record.record_id)
        for module in record.used_by:
            key = f"module:{module}"
            rows.setdefault(
                key,
                {
                    "entity_id": _stable_id("entity", key),
                    "name": module,
                    "entity_type": "module",
                    "source_record_ids": [],
                    "sync_run_id": sync_run_id,
                },
            )["source_record_ids"].append(record.record_id)
    return tuple(rows[key] for key in sorted(rows))


def _import_cypher(gate_plan: GraphDBGatePlan) -> str:
    return "\n".join(
        (
            "// Atomistic Simulation Agent source-to-graph import plan.",
            "// Execute only after the graphdb_write approval gate passes.",
            f"// target_database: {gate_plan.database_name}",
            f"// smoke_query: {gate_plan.smoke_query}",
            "CREATE CONSTRAINT sim_agent_source_url IF NOT EXISTS",
            "FOR (source:SimAgentSourceItem) REQUIRE source.source_url IS UNIQUE;",
            "CREATE CONSTRAINT sim_agent_claim_id IF NOT EXISTS",
            "FOR (claim:PhysicsClaim) REQUIRE claim.record_id IS UNIQUE;",
            "CREATE CONSTRAINT sim_agent_understanding_id IF NOT EXISTS",
            "FOR (understanding:DocumentUnderstanding) REQUIRE understanding.understanding_id IS UNIQUE;",
            "CREATE CONSTRAINT sim_agent_entity_name IF NOT EXISTS",
            "FOR (entity:CanonicalEntity) REQUIRE entity.name IS UNIQUE;",
            "UNWIND $sources AS row",
            "MERGE (source:SimAgentSourceItem {source_url: row.source_url})",
            "SET source += row;",
            "UNWIND $understandings AS row",
            "MATCH (source:SimAgentSourceItem {source_url: row.source_url})",
            "MERGE (understanding:DocumentUnderstanding {understanding_id: row.understanding_id})",
            "SET understanding += row",
            "MERGE (source)-[:HAS_UNDERSTANDING {sync_run_id: row.sync_run_id}]->(understanding);",
            "UNWIND $claims AS row",
            "MATCH (source:SimAgentSourceItem {source_url: row.source_url})",
            "MERGE (claim:PhysicsClaim {record_id: row.record_id})",
            "SET claim += row",
            "MERGE (source)-[:SUPPORTS_CLAIM {sync_run_id: row.sync_run_id}]->(claim);",
            "UNWIND $entities AS row",
            "MERGE (entity:CanonicalEntity {name: row.name})",
            "SET entity += row",
            "WITH entity, row",
            "UNWIND row.source_record_ids AS record_id",
            "MATCH (claim:PhysicsClaim {record_id: record_id})",
            "MERGE (claim)-[:MENTIONS_ENTITY {sync_run_id: row.sync_run_id}]->(entity);",
            "",
        )
    )


def _retrieval_context(gate_plan: GraphDBGatePlan) -> str:
    labels = ", ".join(gate_plan.source_owned_labels)
    relationships = ", ".join(gate_plan.relationships)
    return "\n".join(
        (
            "# Atomistic Simulation Agent Graph Context",
            "",
            f"Database: `{gate_plan.database_name}`",
            f"Database role: `{gate_plan.database_role}`",
            f"Neo4j write enabled: `{str(gate_plan.neo4j_write_enabled).lower()}`",
            f"Smoke query: `{gate_plan.smoke_query}`",
            "",
            "The graph is a derived source catalog. Original papers, run artifacts, and model files remain read-only.",
            f"Source-owned labels: {labels}.",
            f"Relationships: {relationships}.",
            "",
            "Example source-backed lookup:",
            "",
            "```cypher",
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim)",
            "WHERE any(tag IN claim.tags WHERE tag IN $tags)",
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url",
            "ORDER BY claim.confidence DESC",
            "LIMIT 10",
            "```",
            "",
            "Agents must treat low-confidence or missing-evidence claims as review candidates, not accepted physics.",
            "",
        )
    )


def _manifest_payload(gate_plan: GraphDBGatePlan, report: GraphIngestReport) -> dict[str, Any]:
    return {
        "manifest_version": "sim_agent_graph_import_v1",
        "sync_run_id": report.sync_run_id,
        "database_name": gate_plan.database_name,
        "database_role": gate_plan.database_role,
        "requires_empty_database": gate_plan.requires_empty_database,
        "neo4j_write_enabled": gate_plan.neo4j_write_enabled,
        "smoke_query": gate_plan.smoke_query,
        "source_owned_labels": list(gate_plan.source_owned_labels),
        "labels": list(gate_plan.labels),
        "relationships": list(gate_plan.relationships),
        "constraints": list(gate_plan.constraints),
        "conflict_status": gate_plan.conflict_status,
        "ingest_report": report_payload(report),
        "artifacts": {
            "sources": "sources.jsonl",
            "understandings": "understandings.jsonl",
            "claims": "claims.jsonl",
            "canonical_entities": "canonical_entities.jsonl",
            "import_cypher": "import.cypher",
            "retrieval_context": "retrieval_context.md",
            "ingest_report": "ingest_report.json",
        },
    }


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _validated_sync_run_id(sync_run_id: str) -> str:
    value = sync_run_id.strip()
    if not value:
        raise GraphImportError("sync_run_id_required")
    return value


def _stable_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in raw)
    cleaned = "-".join(part for part in cleaned.split("-") if part)[:64]
    if not cleaned:
        cleaned = "item"
    return f"{prefix}-{cleaned}-{digest}"


def _bounded_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _purpose(record: ProvenanceRecord) -> str:
    if "level_set" in record.tags:
        return "profile_evolution_evidence"
    if "force_field" in record.tags:
        return "md_physics_evidence"
    if "surrogate" in record.tags or "mdn" in record.tags:
        return "surrogate_model_evidence"
    if "agents_sdk" in record.tags:
        return "agent_harness_evidence"
    return "source_backed_evidence"


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))
