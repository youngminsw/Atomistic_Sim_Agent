from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_ultraresearch_trace import (
    InsaneSearchTrace,
    UltraresearchArtifactError,
    UltraresearchSource,
    parse_ultraresearch_evidence,
)


ULTRARESEARCH_ACQUISITION_SCHEMA_VERSION: Final = "ultraresearch_acquisition_v1"
ULTRARESEARCH_SOURCE_SCHEMA_VERSION: Final = "ultraresearch_source_v1"
_ARTIFACT_REFS: Final = (
    "ultraresearch/acquisition-plan.json",
    "ultraresearch/research-journal.jsonl",
    "ultraresearch/source-ledger.jsonl",
    "ultraresearch/expansion-log.md",
    "ultraresearch/synthesis-checkpoint.md",
)


@dataclass(frozen=True, slots=True)
class UltraresearchArtifactResult:
    refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UltraresearchArtifactData:
    context: JsonMap
    research_question: str
    source_journal: str
    trace: InsaneSearchTrace
    source_count: int


def materialize_ultraresearch_artifacts(workflow_dir: Path, context: JsonMap, payload: JsonMap) -> UltraresearchArtifactResult:
    evidence = parse_ultraresearch_evidence(payload)
    artifact_data = UltraresearchArtifactData(
        context=context,
        research_question=evidence.research_question,
        source_journal=evidence.source_journal,
        trace=evidence.trace,
        source_count=len(evidence.sources),
    )
    plan_payload = _acquisition_plan(artifact_data)
    journal_rows = _journal_rows(artifact_data)
    source_rows = tuple(_source_row(context, source, index) for index, source in enumerate(evidence.sources, start=1))
    _write_json_once(workflow_dir / "acquisition-plan.json", plan_payload)
    _write_once(workflow_dir / "research-journal.jsonl", _jsonl(journal_rows))
    _write_once(workflow_dir / "source-ledger.jsonl", _jsonl(source_rows))
    _write_once(
        workflow_dir / "expansion-log.md",
        _expansion_log(artifact_data.research_question, artifact_data.trace, artifact_data.source_count),
    )
    _write_once(
        workflow_dir / "synthesis-checkpoint.md",
        _synthesis_checkpoint(artifact_data.source_journal, artifact_data.source_count),
    )
    return UltraresearchArtifactResult(_ARTIFACT_REFS)


def _acquisition_plan(data: UltraresearchArtifactData) -> JsonMap:
    return {
        **data.context,
        "schema_version": ULTRARESEARCH_ACQUISITION_SCHEMA_VERSION,
        "artifact_kind": "ultraresearch_acquisition_plan",
        "research_question": data.research_question,
        "source_journal": data.source_journal,
        "insane_search": {
            "surface": "skill",
            "skill_id": "insane_search",
            "public_only": True,
            "ssrf_safe": True,
            "auth_required": False,
            "ok": data.trace.ok,
            "grid_exhausted": data.trace.grid_exhausted,
            "untried_routes": list(data.trace.untried_routes),
            "must_invoke_playwright_mcp": data.trace.must_invoke_playwright_mcp,
            "stop_reason": data.trace.stop_reason,
            "routes": list(data.trace.routes),
            "source_count": data.source_count,
            "trace": data.trace.raw,
        },
        "content_policy": {
            "untrusted_web_content_is_evidence_only": True,
            "credentialed_or_paywalled_sources_denied": True,
            "private_or_internal_urls_denied": True,
            "citation_required": True,
        },
    }


def _journal_rows(data: UltraresearchArtifactData) -> tuple[JsonMap, ...]:
    return (
        {
            **data.context,
            "artifact_kind": "ultraresearch_question_decomposition",
            "research_question": data.research_question,
            "axes": ["source_discovery", "claim_verification", "synthesis"],
        },
        {
            **data.context,
            "artifact_kind": "ultraresearch_acquisition_wave",
            "route": "insane_search",
            "routes": list(data.trace.routes),
            "source_count": data.source_count,
            "source_journal": data.source_journal,
            "grid_exhausted": data.trace.grid_exhausted,
        },
        {
            **data.context,
            "artifact_kind": "ultraresearch_synthesis_checkpoint",
            "status": "synthesis_ready",
            "citation_required": True,
            "source_ledger": "ultraresearch/source-ledger.jsonl",
        },
    )


def _source_row(context: JsonMap, source: UltraresearchSource, index: int) -> JsonMap:
    return {
        **context,
        "schema_version": ULTRARESEARCH_SOURCE_SCHEMA_VERSION,
        "artifact_kind": "ultraresearch_public_source",
        "source_index": index,
        "url": source.url,
        "route": source.route,
        "title": source.title,
        "evidence_ref": source.evidence_ref,
        "content_trust": "untrusted_evidence",
        "model_instruction_allowed": False,
        "citation_required": True,
    }


def _expansion_log(research_question: str, trace: InsaneSearchTrace, source_count: int) -> str:
    return "\n".join(
        (
            "# Ultraresearch Expansion Log",
            "",
            f"- research_question: {research_question}",
            "- acquisition_backend: insane_search",
            f"- source_count: {source_count}",
            f"- grid_exhausted: {str(trace.grid_exhausted).lower()}",
            f"- untried_routes: {','.join(trace.untried_routes)}",
            f"- must_invoke_playwright_mcp: {str(trace.must_invoke_playwright_mcp).lower()}",
            "- expansion_stop: public_sources_ready_for_cited_synthesis",
            "",
        )
    )


def _synthesis_checkpoint(source_journal: str, source_count: int) -> str:
    return "\n".join(
        (
            "# Ultraresearch Synthesis Checkpoint",
            "",
            "- status: synthesis_ready",
            "- citation_required: true",
            "- content_trust: untrusted_evidence_only",
            f"- source_journal: {source_journal}",
            f"- source_count: {source_count}",
            "",
        )
    )


def _jsonl(rows: tuple[JsonMap, ...]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _write_json_once(path: Path, payload: JsonMap) -> None:
    _write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_once(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise UltraresearchArtifactError("ultraresearch_artifact_corrupt") from exc
        if current != body:
            raise UltraresearchArtifactError("ultraresearch_artifact_conflict")
        return
    path.write_text(body, encoding="utf-8")
