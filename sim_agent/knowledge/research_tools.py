from __future__ import annotations

from dataclasses import dataclass

from .provenance_registry import ProvenanceRegistry
from .types import GraphDBDryRunBundle, KnowledgeRegistryError, ProvenanceRecord, SourceKind


class ResearchToolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Citation:
    record_id: str
    source_url: str
    title: str


@dataclass(frozen=True, slots=True)
class ResearchImportRequest:
    source_url: str
    title: str
    claim: str
    tags: tuple[str, ...]
    used_by: tuple[str, ...]
    source_kind: SourceKind = SourceKind.PAPER
    confidence: float = 0.7
    record_id: str = ""
    extracted_on: str = "2026-06-10"


@dataclass(frozen=True, slots=True)
class ResearchImportResult:
    provenance_ready: bool
    graphdb_write: bool
    imported_record: ProvenanceRecord
    registry: ProvenanceRegistry
    graphdb_bundle: GraphDBDryRunBundle


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    query: str
    tags: tuple[str, ...]
    max_summary_chars: int = 480


@dataclass(frozen=True, slots=True)
class ResearchAnswer:
    answer_status: str
    summary: str
    citations: tuple[Citation, ...]
    evidence_record_ids: tuple[str, ...]
    graph_lookup_query: str


def import_research_source(registry: ProvenanceRegistry, request: ResearchImportRequest) -> ResearchImportResult:
    record = ProvenanceRecord(
        record_id=request.record_id or _record_id(request.source_url, request.title),
        source_url=request.source_url,
        title=request.title,
        claim=request.claim,
        tags=request.tags,
        confidence=request.confidence,
        extracted_on=request.extracted_on,
        used_by=request.used_by,
        source_kind=request.source_kind,
    )
    try:
        updated = registry.with_record(record)
    except KnowledgeRegistryError as exc:
        raise ResearchToolError(str(exc)) from exc
    return ResearchImportResult(
        provenance_ready=True,
        graphdb_write=False,
        imported_record=record,
        registry=updated,
        graphdb_bundle=updated.export_graphdb_dry_run(),
    )


def answer_research_question(registry: ProvenanceRegistry, question: ResearchQuestion) -> ResearchAnswer:
    records = _matching_records(registry, question)
    if not records:
        return ResearchAnswer(
            answer_status="missing_evidence",
            summary="No source-backed record matched the requested evidence.",
            citations=(),
            evidence_record_ids=(),
            graph_lookup_query=_graph_lookup_query(question.tags),
        )
    return ResearchAnswer(
        answer_status="answered",
        summary=_summary(records, question.max_summary_chars),
        citations=tuple(Citation(record.record_id, record.source_url, record.title) for record in records),
        evidence_record_ids=tuple(record.record_id for record in records),
        graph_lookup_query=_graph_lookup_query(question.tags),
    )


def _matching_records(registry: ProvenanceRegistry, question: ResearchQuestion) -> tuple[ProvenanceRecord, ...]:
    tagged = tuple(record for record in registry.records if _tag_match(record, question.tags))
    if tagged:
        return tagged
    query_terms = tuple(term.lower() for term in question.query.split() if len(term) > 3)
    return tuple(record for record in registry.records if _claim_match(record, query_terms))


def _tag_match(record: ProvenanceRecord, tags: tuple[str, ...]) -> bool:
    return bool(tags) and any(tag in record.tags for tag in tags)


def _claim_match(record: ProvenanceRecord, query_terms: tuple[str, ...]) -> bool:
    claim = record.claim.lower()
    title = record.title.lower()
    return any(term in claim or term in title for term in query_terms)


def _summary(records: tuple[ProvenanceRecord, ...], max_chars: int) -> str:
    first = records[0]
    summary = f"{first.claim} Citation count: {len(records)}."
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."


def _graph_lookup_query(tags: tuple[str, ...]) -> str:
    if tags:
        return (
            "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
            "WHERE any(tag IN claim.tags WHERE tag IN $tags) "
            "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url"
        )
    return (
        "MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim) "
        "RETURN claim.record_id, claim.claim, claim.confidence, source.source_url"
    )


def _record_id(source_url: str, title: str) -> str:
    raw = title or source_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in raw)
    return "-".join(part for part in cleaned.split("-") if part)
