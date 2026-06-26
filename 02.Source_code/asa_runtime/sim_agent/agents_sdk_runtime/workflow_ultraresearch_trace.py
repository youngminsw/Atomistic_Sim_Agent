from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from sim_agent.schemas._parse import JsonMap

from .workflow_harness_payload import text_value


_CREDENTIAL_STOP_MARKERS: Final = ("auth", "credential", "paywall", "login")
JsonEvidenceValue = JsonMap | list[JsonMap | str | int | float | bool | None] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class UltraresearchArtifactError(Exception):
    blocker: str

    def __str__(self) -> str:
        return self.blocker


@dataclass(frozen=True, slots=True)
class InsaneSearchTrace:
    raw: JsonMap
    ok: bool
    grid_exhausted: bool
    untried_routes: tuple[str, ...]
    must_invoke_playwright_mcp: bool
    stop_reason: str
    routes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UltraresearchSource:
    url: str
    route: str
    title: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class ParsedUltraresearchEvidence:
    research_question: str
    source_journal: str
    trace: InsaneSearchTrace
    sources: tuple[UltraresearchSource, ...]


def parse_ultraresearch_evidence(payload: JsonMap) -> ParsedUltraresearchEvidence:
    evidence = _evidence(payload)
    research_question = text_value(evidence.get("research_question"), "")
    source_journal = text_value(evidence.get("source_journal"), "")
    if not research_question:
        raise UltraresearchArtifactError("ultraresearch_question_required")
    if not source_journal:
        raise UltraresearchArtifactError("ultraresearch_source_journal_required")
    trace = _insane_search_trace(evidence.get("insane_search_trace"))
    _block_if_trace_not_usable(trace)
    sources = _sources(trace.raw)
    if not sources:
        raise UltraresearchArtifactError("ultraresearch_public_sources_required")
    return ParsedUltraresearchEvidence(research_question, source_journal, trace, sources)


def _evidence(payload: JsonMap) -> JsonMap:
    value = payload.get("evidence")
    if isinstance(value, dict):
        return value
    return {}


def _insane_search_trace(value: JsonEvidenceValue) -> InsaneSearchTrace:
    loaded = value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UltraresearchArtifactError("ultraresearch_insane_search_trace_invalid") from exc
    if not isinstance(loaded, dict):
        raise UltraresearchArtifactError("ultraresearch_insane_search_trace_invalid")
    if text_value(loaded.get("skill_id"), "") != "insane_search":
        raise UltraresearchArtifactError("ultraresearch_insane_search_trace_invalid")
    if text_value(loaded.get("surface"), "skill") != "skill":
        raise UltraresearchArtifactError("ultraresearch_insane_search_trace_invalid")
    return InsaneSearchTrace(
        raw=loaded,
        ok=_bool_field(loaded, "ok"),
        grid_exhausted=_bool_field(loaded, "grid_exhausted"),
        untried_routes=_str_tuple(loaded.get("untried_routes")),
        must_invoke_playwright_mcp=_bool_field(loaded, "must_invoke_playwright_mcp"),
        stop_reason=text_value(loaded.get("stop_reason"), ""),
        routes=_str_tuple(loaded.get("routes")),
    )


def _block_if_trace_not_usable(trace: InsaneSearchTrace) -> None:
    if _credentialed(trace):
        raise UltraresearchArtifactError("ultraresearch_credentialed_source_denied")
    if _bool_field(trace.raw, "public_only") is not True or _bool_field(trace.raw, "ssrf_safe") is not True:
        raise UltraresearchArtifactError("ultraresearch_public_boundary_required")
    if not trace.ok and (not trace.grid_exhausted or trace.untried_routes or trace.must_invoke_playwright_mcp):
        raise UltraresearchArtifactError("ultraresearch_acquisition_not_exhausted")
    if not trace.ok:
        raise UltraresearchArtifactError("ultraresearch_public_sources_required")


def _credentialed(trace: InsaneSearchTrace) -> bool:
    if _optional_bool(trace.raw, "auth_required") or _optional_bool(trace.raw, "paywall_required"):
        return True
    stop_reason = trace.stop_reason.casefold()
    return any(marker in stop_reason for marker in _CREDENTIAL_STOP_MARKERS)


def _sources(trace: JsonMap) -> tuple[UltraresearchSource, ...]:
    value = trace.get("sources")
    if not isinstance(value, list | tuple):
        return ()
    sources: list[UltraresearchSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise UltraresearchArtifactError("ultraresearch_source_trace_invalid")
        url = text_value(item.get("url"), "")
        if not _is_public_http_url(url):
            raise UltraresearchArtifactError("insane_search_public_content_only")
        sources.append(
            UltraresearchSource(
                url=url,
                route=text_value(item.get("route"), "unknown"),
                title=text_value(item.get("title"), "untitled public evidence"),
                evidence_ref=text_value(item.get("evidence_ref"), "insane_search_trace"),
            )
        )
    return tuple(sources)


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    lowered = host.casefold()
    if lowered in {"localhost", "metadata.google.internal"} or lowered.endswith((".local", ".internal")):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _bool_field(mapping: JsonMap, field: str) -> bool:
    value = mapping.get(field)
    if isinstance(value, bool):
        return value
    raise UltraresearchArtifactError("ultraresearch_insane_search_trace_invalid")


def _optional_bool(mapping: JsonMap, field: str) -> bool:
    value = mapping.get(field)
    return value if isinstance(value, bool) else False


def _str_tuple(value: JsonEvidenceValue) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)
