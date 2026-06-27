from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .workflow_evidence_hashes import artifact_hashes, verify_artifact_hash


type JsonValue = str | int | float | bool | None | JsonMap | list["JsonValue"] | tuple["JsonValue", ...]

MANIFEST_SCHEMA_VERSION: Final = "asa_workflow_gap_evidence_manifest_v1"
PARITY_MATRIX_NAME: Final = "gajae_workflow_parity"
EVIDENCE_PREFIX: Final = ".omo/evidence/asa-gajae-workflow-gap-closure/"
FAKE_LIVE_PROVIDERS: Final = frozenset({"fake", "mock", "static", "fixture", "replay", "cassette"})


@dataclass(frozen=True, slots=True)
class WorkflowGapEvidenceResult:
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def check_workflow_gap_evidence(manifest_path: Path, evidence_dir: Path, parity_path: Path) -> WorkflowGapEvidenceResult:
    blockers: list[str] = []
    _verify_parity(_parity_json_path(parity_path, evidence_dir), evidence_dir, blockers)
    _verify_manifest(manifest_path, evidence_dir, blockers)
    return WorkflowGapEvidenceResult(tuple(dict.fromkeys(blockers)))


def _parity_json_path(parity_path: Path, evidence_dir: Path) -> Path:
    if parity_path.suffix != ".md":
        return parity_path
    fixture_parity = evidence_dir / "parity.json"
    if fixture_parity.is_file():
        return fixture_parity
    return parity_path.with_suffix(".json")


def _verify_parity(parity_path: Path, evidence_dir: Path, blockers: list[str]) -> None:
    parity = _read_json(parity_path, blockers, "parity_json_unreadable")
    if parity is None:
        return
    _require_equal(parity, "matrix", PARITY_MATRIX_NAME, blockers, "parity_matrix_unknown")
    rows = _sequence_field(parity, "rows", blockers, "parity_rows_missing")
    for index, raw_row in enumerate(rows):
        row = _as_mapping(raw_row, f"rows[{index}]", blockers, "parity_row_invalid")
        if row is None:
            continue
        row_id = _required_str(row, "id", blockers, "parity_row_id_missing")
        status = _required_str(row, "status", blockers, "parity_status_missing")
        if row_id is None or status != "implemented":
            continue
        _verify_parity_row(row, evidence_dir, blockers)


def _verify_parity_row(row: JsonMap, evidence_dir: Path, blockers: list[str]) -> None:
    for field in ("gajae_reference", "behavioral_contract", "asa_target", "tests", "verification_evidence"):
        if not _has_required_field(row, field):
            blockers.append(f"parity_{field}_missing")
    anti_false_green = _as_mapping(row.get("anti_false_green"), "anti_false_green", blockers, "parity_negative_control_missing")
    if anti_false_green is not None:
        for field in ("sabotage", "detected_by", "blocker"):
            if not _has_required_field(anti_false_green, field):
                blockers.append("parity_negative_control_missing")
    evidence_items = _sequence_field(row, "verification_evidence", blockers, "parity_evidence_missing")
    for raw_path in evidence_items:
        evidence_path = _path_text(raw_path, blockers, "parity_evidence_missing")
        if evidence_path is None:
            continue
        if not _resolve_evidence_path(evidence_dir, evidence_path).is_file():
            blockers.append("parity_evidence_missing")


def _verify_manifest(manifest_path: Path, evidence_dir: Path, blockers: list[str]) -> None:
    manifest = _read_json(manifest_path, blockers, "manifest_json_unreadable")
    if manifest is None:
        return
    _require_equal(manifest, "schema_version", MANIFEST_SCHEMA_VERSION, blockers, "manifest_schema_unknown")
    run_id = _required_str(manifest, "run_id", blockers, "manifest_run_id_missing")
    if run_id is None:
        return
    _verify_manifest_evidence(manifest, evidence_dir, run_id, blockers)
    _verify_e2e_scenarios(manifest, evidence_dir, run_id, blockers)
    _verify_sabotage_cases(manifest, blockers)
    _verify_live_rows(manifest, evidence_dir, blockers)


def _verify_manifest_evidence(manifest: JsonMap, evidence_dir: Path, run_id: str, blockers: list[str]) -> None:
    for index, entry in enumerate(_sequence_field(manifest, "evidence_files", blockers, "manifest_evidence_missing")):
        evidence = _as_mapping(entry, f"evidence_files[{index}]", blockers, "manifest_evidence_missing")
        if evidence is None:
            continue
        evidence_path = _entry_path(evidence, "path", evidence_dir, blockers, "manifest_evidence_missing")
        text = _read_text(evidence_path, blockers, "manifest_evidence_missing") if evidence_path is not None else None
        if text is None:
            continue
        required_tokens = [run_id, *_string_items(evidence, "contains", blockers, "manifest_evidence_token_missing")]
        for token in required_tokens:
            if token not in text:
                blockers.append("manifest_evidence_token_missing")


def _verify_e2e_scenarios(manifest: JsonMap, evidence_dir: Path, run_id: str, blockers: list[str]) -> None:
    for index, entry in enumerate(_sequence_field(manifest, "e2e_scenarios", blockers, "e2e_scenario_missing")):
        scenario = _as_mapping(entry, f"e2e_scenarios[{index}]", blockers, "e2e_scenario_missing")
        if scenario is None:
            continue
        _verify_e2e_scenario(scenario, evidence_dir, run_id, blockers)


def _verify_e2e_scenario(scenario: JsonMap, evidence_dir: Path, run_id: str, blockers: list[str]) -> None:
    ledger_path = _entry_path(scenario, "json", evidence_dir, blockers, "e2e_ledger_missing")
    transcript_path = _entry_path(scenario, "transcript", evidence_dir, blockers, "e2e_transcript_missing")
    ledger = _read_json(ledger_path, blockers, "e2e_ledger_missing") if ledger_path is not None else None
    transcript = _read_text(transcript_path, blockers, "e2e_transcript_missing") if transcript_path is not None else None
    if ledger is not None:
        _require_equal(ledger, "run_id", run_id, blockers, "e2e_run_id_mismatch")
        _require_equal(ledger, "status", "succeeded", blockers, "e2e_status_mismatch")
    if transcript is not None:
        if run_id not in transcript:
            blockers.append("e2e_run_id_mismatch")
        for token in _string_items(scenario, "required_transcript", blockers, "e2e_transcript_missing_required"):
            if token not in transcript:
                blockers.append("e2e_transcript_missing_required")
    _verify_artifacts(scenario, ledger, evidence_dir, run_id, blockers)


def _verify_artifacts(
    scenario: JsonMap,
    ledger: JsonMap | None,
    evidence_dir: Path,
    run_id: str,
    blockers: list[str],
) -> None:
    started_at = _optional_float(scenario, "started_at_epoch", blockers, "artifact_started_at_invalid")
    hashes = artifact_hashes(ledger)
    for raw_path in _sequence_field(scenario, "artifacts", blockers, "artifact_missing"):
        artifact_text = _path_text(raw_path, blockers, "artifact_missing")
        if artifact_text is None:
            continue
        artifact_path = _resolve_evidence_path(evidence_dir, artifact_text)
        body = _read_text(artifact_path, blockers, "artifact_missing")
        if body is None:
            continue
        if run_id not in body:
            blockers.append("artifact_run_id_mismatch")
        if started_at is not None and artifact_path.stat().st_mtime < started_at:
            blockers.append("artifact_stale")
        verify_artifact_hash(artifact_path, artifact_text, hashes, blockers)


def _verify_sabotage_cases(manifest: JsonMap, blockers: list[str]) -> None:
    cases = _sequence_field(manifest, "sabotage_cases", blockers, "sabotage_case_missing")
    for index, entry in enumerate(cases):
        sabotage = _as_mapping(entry, f"sabotage_cases[{index}]", blockers, "sabotage_case_missing")
        if sabotage is None:
            continue
        _require_present(sabotage, "case", blockers, "sabotage_case_missing")
        _require_present(sabotage, "blocker", blockers, "sabotage_case_missing")
        _require_equal(sabotage, "status", "detected", blockers, "sabotage_case_not_detected")


def _verify_live_rows(manifest: JsonMap, evidence_dir: Path, blockers: list[str]) -> None:
    rows = _sequence_field(manifest, "live_llm", blockers, "live_llm_evidence_missing")
    for index, entry in enumerate(rows):
        live_row = _as_mapping(entry, f"live_llm[{index}]", blockers, "live_llm_evidence_missing")
        if live_row is None:
            continue
        _verify_live_row(live_row, evidence_dir, blockers)


def _verify_live_row(live_row: JsonMap, evidence_dir: Path, blockers: list[str]) -> None:
    status = _required_str(live_row, "status", blockers, "live_llm_status_missing")
    if status == "blocked" and live_row.get("blocker") == "live_llm_provider_unavailable":
        blockers.append("live_llm_provider_unavailable")
        return
    _require_equal(live_row, "status", "succeeded", blockers, "live_llm_status_missing")
    provider = _required_str(live_row, "provider", blockers, "live_llm_provider_missing")
    if provider in FAKE_LIVE_PROVIDERS:
        blockers.append("live_llm_fake_provider")
    request_id = _required_str(live_row, "provider_request_id", blockers, "live_llm_provider_event_missing")
    assistant_id = _required_str(live_row, "assistant_message_id", blockers, "live_llm_provider_event_missing")
    event_path = _entry_path(live_row, "provider_events", evidence_dir, blockers, "live_llm_provider_event_missing")
    event_text = _read_text(event_path, blockers, "live_llm_provider_event_missing") if event_path is not None else None
    if event_text is None:
        return
    if request_id is not None and request_id not in event_text:
        blockers.append("live_llm_provider_event_missing")
    if assistant_id is not None and assistant_id not in event_text:
        blockers.append("live_llm_provider_event_missing")


def _read_json(path: Path, blockers: list[str], blocker: str) -> JsonMap | None:
    try:
        return as_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SchemaValidationError):
        blockers.append(blocker)
        return None


def _read_text(path: Path, blockers: list[str], blocker: str) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        blockers.append(blocker)
        return None


def _entry_path(entry: JsonMap, field: str, evidence_dir: Path, blockers: list[str], blocker: str) -> Path | None:
    path_text = _required_str(entry, field, blockers, blocker)
    return _resolve_evidence_path(evidence_dir, path_text) if path_text is not None else None


def _resolve_evidence_path(evidence_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith(EVIDENCE_PREFIX):
        return evidence_dir / path_text.removeprefix(EVIDENCE_PREFIX)
    return evidence_dir / path_text


def _required_str(mapping: JsonMap, field: str, blockers: list[str], blocker: str) -> str | None:
    try:
        return as_str(require(mapping, field), field)
    except SchemaValidationError:
        blockers.append(blocker)
        return None


def _optional_float(mapping: JsonMap, field: str, blockers: list[str], blocker: str) -> float | None:
    value = mapping.get(field)
    if value is None:
        return None
    try:
        return as_float(value, field)
    except SchemaValidationError:
        blockers.append(blocker)
        return None


def _sequence_field(mapping: JsonMap, field: str, blockers: list[str], blocker: str) -> tuple[JsonValue, ...]:
    try:
        return tuple(as_sequence(require(mapping, field), field))
    except SchemaValidationError:
        blockers.append(blocker)
        return ()


def _string_items(mapping: JsonMap, field: str, blockers: list[str], blocker: str) -> tuple[str, ...]:
    values = _sequence_field(mapping, field, blockers, blocker)
    parsed: list[str] = []
    for value in values:
        parsed_value = _path_text(value, blockers, blocker)
        if parsed_value is not None:
            parsed.append(parsed_value)
    return tuple(parsed)


def _path_text(value: JsonValue, blockers: list[str], blocker: str) -> str | None:
    try:
        return as_str(value, "path")
    except SchemaValidationError:
        blockers.append(blocker)
        return None


def _as_mapping(value: JsonValue, field: str, blockers: list[str], blocker: str) -> JsonMap | None:
    try:
        return as_mapping(value, field)
    except SchemaValidationError:
        blockers.append(blocker)
        return None


def _require_equal(mapping: JsonMap, field: str, expected: str, blockers: list[str], blocker: str) -> None:
    if mapping.get(field) != expected:
        blockers.append(blocker)


def _require_present(mapping: JsonMap, field: str, blockers: list[str], blocker: str) -> None:
    if not _has_required_field(mapping, field):
        blockers.append(blocker)


def _has_required_field(mapping: JsonMap, field: str) -> bool:
    value = mapping.get(field)
    if value is None:
        return False
    if isinstance(value, str | list | tuple):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    return True
