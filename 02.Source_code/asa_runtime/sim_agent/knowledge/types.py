from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceKind(StrEnum):
    CODE_SAMPLE = "code_sample"
    DOCS = "docs"
    PAPER = "paper"
    POLICY = "policy"
    POTENTIAL = "potential"
    REPOSITORY = "repository"


class KnowledgeRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    record_id: str
    source_url: str
    title: str
    claim: str
    tags: tuple[str, ...]
    confidence: float
    extracted_on: str
    used_by: tuple[str, ...]
    source_kind: SourceKind


@dataclass(frozen=True, slots=True)
class GraphDBDryRunBundle:
    neo4j_write_enabled: bool
    database_name: str
    database_role: str
    requires_empty_database: bool
    labels: tuple[str, ...]
    relationships: tuple[str, ...]
    constraints: tuple[str, ...]
    records: tuple[ProvenanceRecord, ...]

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"neo4j_write_enabled={str(self.neo4j_write_enabled).lower()}",
            f"database_name={self.database_name}",
            f"database_role={self.database_role}",
            f"requires_empty_database={str(self.requires_empty_database).lower()}",
            f"labels={','.join(self.labels)}",
            f"relationships={','.join(self.relationships)}",
            f"record_count={len(self.records)}",
        )
