from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .graph_import import GraphImportBundle, build_source_graph_import_bundle
from .graphdb_gate import GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan
from .provenance_registry import ProvenanceRegistry, seeded_provenance_registry
from .types import ProvenanceRecord, SourceKind


class MemorySeedError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MemorySeedSource:
    source_url: str
    title: str
    claim: str
    tags: tuple[str, ...]
    used_by: tuple[str, ...]
    confidence: float
    source_kind: SourceKind


DEFAULT_MEMORY_TERMS: tuple[str, ...] = (
    "MD simulation",
    "LAMMPS",
    "plasma etching",
    "etching",
    "MDN",
    "surrogate",
    "Level Set",
)


def build_memory_seed_bundle(
    output_dir: Path,
    *,
    database_name: str,
    sync_run_id: str,
    memory_sources: Sequence[MemorySeedSource],
) -> GraphImportBundle:
    registry = seeded_provenance_registry()
    for source in memory_sources:
        registry = registry.with_record(_record_from_source(source))
    gate_plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=(),
            database_name=database_name,
            requires_empty_database=True,
        )
    )
    return build_source_graph_import_bundle(registry, gate_plan, output_dir, sync_run_id=sync_run_id)


def read_memory_seed_sources_from_neo4j(
    *,
    terms: Sequence[str] = DEFAULT_MEMORY_TERMS,
    limit_per_term: int = 4,
    uri_env: str = "NEO4J_URI",
    user_env: str = "NEO4J_USERNAME",
    password_env: str = "NEO4J_PASSWORD",
    database_env: str = "NEO4J_DATABASE",
) -> tuple[MemorySeedSource, ...]:
    uri = _env(uri_env)
    username = _env(user_env)
    password = _env(password_env)
    database = os.environ.get(database_env, "neo4j")
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import Neo4jError, ServiceUnavailable
    except ImportError as exc:
        raise MemorySeedError("neo4j_driver_not_installed") from exc
    try:
        rows: list[JsonMap] = []
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            with driver.session(database=database) as session:
                session.run("RETURN 1 AS ok").single()
                for term in terms:
                    rows.extend(
                        dict(record)
                        for record in session.run(_SOURCE_MEMORY_QUERY, term=term, limit=limit_per_term)
                    )
    except (Neo4jError, ServiceUnavailable) as exc:
        raise MemorySeedError(exc.__class__.__name__) from exc
    return memory_sources_from_rows(rows)


def memory_sources_from_rows(rows: Iterable[Mapping[str, object]]) -> tuple[MemorySeedSource, ...]:
    sources: dict[str, MemorySeedSource] = {}
    for row in rows:
        source = _source_from_row(row)
        if source is None or source.source_url in sources:
            continue
        sources[source.source_url] = source
    return tuple(sources[key] for key in sorted(sources))


def _source_from_row(row: Mapping[str, object]) -> MemorySeedSource | None:
    source_url = _first_text(row, ("source_uri", "page_url", "source_path", "source"))
    title = _first_text(row, ("title", "filename", "concept")) or "Memory evidence"
    summary = _first_text(row, ("summary", "evidence_excerpt", "relation_evidence"))
    if not source_url or not summary:
        return None
    tags = _tags(f"{title} {summary}")
    return MemorySeedSource(
        source_url=source_url,
        title=title[:180],
        claim=summary[:420],
        tags=tags,
        used_by=_used_by(tags),
        confidence=_confidence(row.get("confidence")),
        source_kind=_source_kind(source_url),
    )


def _record_from_source(source: MemorySeedSource) -> ProvenanceRecord:
    digest = hashlib.sha1(source.source_url.encode("utf-8")).hexdigest()[:10]
    return ProvenanceRecord(
        record_id=f"memory-seed-{_slug(source.title)}-{digest}"[:96],
        source_url=source.source_url,
        title=source.title,
        claim=source.claim,
        tags=source.tags,
        confidence=source.confidence,
        extracted_on="2026-06-18",
        used_by=source.used_by,
        source_kind=source.source_kind,
    )


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise MemorySeedError(f"missing_env:{name}")


def _first_text(row: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.68


def _tags(text: str) -> tuple[str, ...]:
    lower = text.lower()
    tags: list[str] = ["memory_seed"]
    for needle, tag in (
        ("lammps", "lammps"),
        ("md simulation", "md"),
        ("molecular dynamics", "md"),
        ("plasma etch", "etching"),
        ("etching", "etching"),
        ("mdn", "mdn"),
        ("surrogate", "surrogate"),
        ("level set", "level_set"),
        ("force field", "force_field"),
    ):
        if needle in lower and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def _used_by(tags: Sequence[str]) -> tuple[str, ...]:
    users = ["research_agent", "qa_agent"]
    if "md" in tags or "lammps" in tags or "force_field" in tags:
        users.append("md_agent")
    if "mdn" in tags or "surrogate" in tags:
        users.append("ml_agent")
    if "level_set" in tags or "etching" in tags:
        users.append("feature_scale_agent")
    return tuple(dict.fromkeys(users))


def _source_kind(source_url: str) -> SourceKind:
    if source_url.startswith("http"):
        return SourceKind.PAPER
    if source_url.startswith("/"):
        return SourceKind.REPOSITORY
    return SourceKind.POLICY


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "source"


_SOURCE_MEMORY_QUERY = """
MATCH (memory:SourceMemory)
WHERE coalesce(memory.import_status, "active") = "active"
  AND (
    toLower(coalesce(memory.title, "")) CONTAINS toLower($term)
    OR toLower(coalesce(memory.summary, "")) CONTAINS toLower($term)
    OR toLower(coalesce(memory.source_path, "")) CONTAINS toLower($term)
    OR toLower(coalesce(memory.page_url, "")) CONTAINS toLower($term)
  )
OPTIONAL MATCH (memory)-[supportedBy:SUPPORTED_BY]->(evidence:SourceEvidence)
WHERE coalesce(supportedBy.import_status, "active") = "active"
  AND coalesce(evidence.import_status, "active") = "active"
OPTIONAL MATCH (memory)-[derivedFrom:DERIVED_FROM]->(source)
WHERE coalesce(derivedFrom.import_status, "active") = "active"
RETURN memory.title AS title,
       memory.summary AS summary,
       memory.confidence AS confidence,
       coalesce(source.source_path, memory.source_path) AS source_path,
       coalesce(source.source_uri, memory.source_uri) AS source_uri,
       coalesce(source.page_url, memory.page_url) AS page_url,
       evidence.excerpt AS evidence_excerpt
ORDER BY CASE memory.importance
  WHEN "critical" THEN 0
  WHEN "high" THEN 1
  WHEN "medium" THEN 2
  ELSE 3
END, memory.confidence DESC
LIMIT $limit
"""
