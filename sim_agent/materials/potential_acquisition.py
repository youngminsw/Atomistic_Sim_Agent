from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str
from sim_agent.schemas.errors import SchemaValidationError

from .potential_gate import PotentialValidationReport, validate_potential_candidate
from .potential_sandbox import (
    PotentialSandboxSmokeReport,
    PotentialSandboxSmokeRequest,
    run_potential_sandbox_smoke,
)


MAX_POTENTIAL_BYTES: Final = 2_000_000
LEGACY_SOURCE_PREFIX: Final = "02.Source_code/mss_agent/"


@dataclass(frozen=True, slots=True)
class PotentialAcquisitionError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class PotentialAcquisitionRequest:
    source_url: str
    metadata_url: str
    material_id: str
    ion_species: str
    required_elements: tuple[str, ...]
    timeout_s: float = 20.0
    sandbox_command: tuple[str, ...] = ()
    sandbox_work_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class PotentialAcquisitionReport:
    ok: bool
    candidate_payload: JsonMap
    validation: PotentialValidationReport
    payload: JsonMap


@dataclass(frozen=True, slots=True)
class FetchedText:
    url: str
    text: str
    sha256: str
    byte_count: int


def acquire_potential_candidate(
    request: PotentialAcquisitionRequest,
    repo_root: Path,
) -> PotentialAcquisitionReport:
    evidence: list[str] = []
    potential = _fetch_text(request.source_url, repo_root, request.timeout_s)
    metadata = _read_metadata(request.metadata_url, repo_root, request.timeout_s)
    evidence.append("potential_file_fetched")
    evidence.append("metadata_file_fetched")

    parsed_elements = _extract_elements(potential.text)
    element_symbols = parsed_elements or _metadata_tuple(metadata, "element_symbols")
    if element_symbols:
        evidence.append("format_parse_passed")
    else:
        evidence.append("format_parse_missing_elements")

    draft_candidate = _candidate_payload(request, metadata, potential, element_symbols, False)
    sandbox = _sandbox_smoke(request, repo_root, potential.text, draft_candidate)
    evidence.extend(_sandbox_evidence(sandbox))
    candidate = _candidate_payload(request, metadata, potential, element_symbols, sandbox.ok)
    validation = validate_potential_candidate(
        candidate,
        material_id=request.material_id,
        ion_species=request.ion_species,
        required_elements=request.required_elements,
    )
    gate_status = _payload_text(validation.payload, "gate_status")
    ok = validation.ok
    return PotentialAcquisitionReport(
        ok=ok,
        candidate_payload=candidate,
        validation=validation,
        payload={
            "ok": ok,
            "gate_status": gate_status,
            "candidate": candidate,
            "validation": validation.payload,
            "sandbox_smoke": sandbox.payload,
            "acquisition_evidence": evidence,
            "acquisition_errors": sandbox.payload["errors"],
        },
    )


def _sandbox_smoke(
    request: PotentialAcquisitionRequest,
    repo_root: Path,
    potential_text: str,
    candidate: JsonMap,
) -> PotentialSandboxSmokeReport:
    work_dir = request.sandbox_work_dir or repo_root / ".omo" / "potential-smoke"
    return run_potential_sandbox_smoke(
        PotentialSandboxSmokeRequest(
            candidate_payload=candidate,
            potential_text=potential_text,
            work_dir=work_dir,
            lammps_command=request.sandbox_command,
            timeout_s=request.timeout_s,
        )
    )


def _read_metadata(metadata_url: str, repo_root: Path, timeout_s: float) -> JsonMap:
    fetched = _fetch_text(metadata_url, repo_root, timeout_s)
    try:
        return as_mapping(json.loads(fetched.text), "potential_metadata")
    except json.JSONDecodeError as exc:
        raise PotentialAcquisitionError("metadata_json_invalid") from exc


def _fetch_text(resource_url: str, repo_root: Path, timeout_s: float) -> FetchedText:
    if resource_url.startswith("repo://"):
        return _fetch_local_text(_repo_path(resource_url, repo_root), resource_url)
    if resource_url.startswith("file://"):
        return _fetch_local_text(_file_uri_path(resource_url), resource_url)
    if resource_url.startswith(("https://", "http://")):
        return _fetch_http_text(resource_url, timeout_s)
    raise PotentialAcquisitionError("unsupported_resource_url")


def _fetch_local_text(path: Path, resource_url: str) -> FetchedText:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PotentialAcquisitionError("potential_resource_not_found") from exc
    return _fetched_text(resource_url, raw)


def _fetch_http_text(resource_url: str, timeout_s: float) -> FetchedText:
    request = Request(resource_url, headers={"User-Agent": "atomistic-sim-agent/0.1"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read(MAX_POTENTIAL_BYTES + 1)
    except HTTPError as exc:
        raise PotentialAcquisitionError(f"http_status_{exc.code}") from exc
    except URLError as exc:
        raise PotentialAcquisitionError("http_fetch_failed") from exc
    except TimeoutError as exc:
        raise PotentialAcquisitionError("http_fetch_timeout") from exc
    return _fetched_text(resource_url, raw)


def _fetched_text(resource_url: str, raw: bytes) -> FetchedText:
    if len(raw) > MAX_POTENTIAL_BYTES:
        raise PotentialAcquisitionError("potential_file_too_large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PotentialAcquisitionError("potential_file_not_utf8") from exc
    return FetchedText(
        url=resource_url,
        text=text,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def _candidate_payload(
    request: PotentialAcquisitionRequest,
    metadata: JsonMap,
    potential: FetchedText,
    element_symbols: tuple[str, ...],
    syntax_smoke_passed: bool,
) -> JsonMap:
    return {
        "potential_id": _metadata_text(metadata, "potential_id"),
        "material_id": _metadata_text(metadata, "material_id"),
        "ion_species": _metadata_text(metadata, "ion_species"),
        "pair_style": _metadata_text(metadata, "pair_style"),
        "potential_name": _metadata_text(metadata, "potential_name"),
        "source_url": potential.url,
        "source_sha256": potential.sha256,
        "source_bytes": potential.byte_count,
        "provenance_url": _metadata_text(metadata, "provenance_url"),
        "publication_url": _metadata_text(metadata, "publication_url"),
        "publication_doi": _metadata_text(metadata, "publication_doi"),
        "license": _metadata_text(metadata, "license"),
        "lammps_unit_style": _metadata_text(metadata, "lammps_unit_style"),
        "element_symbols": list(element_symbols),
        "atom_type_mapping": list(_metadata_tuple(metadata, "atom_type_mapping")),
        "syntax_smoke_passed": syntax_smoke_passed,
        "fitted_system": _metadata_text(metadata, "fitted_system"),
        "transferability_scope": _metadata_text(metadata, "transferability_scope"),
        "acquisition_material_id": request.material_id,
        "acquisition_ion_species": request.ion_species,
    }


def _extract_elements(potential_text: str) -> tuple[str, ...]:
    for line in potential_text.splitlines():
        normalized = line.strip()
        if normalized.lower().startswith("# elements:"):
            return _split_symbols(normalized.split(":", maxsplit=1)[1])
    return ()


def _metadata_text(metadata: JsonMap, field: str) -> str:
    value = metadata.get(field)
    if isinstance(value, str):
        return value
    return ""


def _metadata_tuple(metadata: JsonMap, field: str) -> tuple[str, ...]:
    value = metadata.get(field)
    if value is None:
        return ()
    try:
        return tuple(as_str(item, field) for item in as_sequence(value, field))
    except SchemaValidationError:
        return ()


def _payload_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def _sandbox_evidence(sandbox: PotentialSandboxSmokeReport) -> list[str]:
    status = _payload_text(sandbox.payload, "smoke_status")
    if sandbox.ok:
        return [status, "lammps_sandbox_smoke_passed"]
    return [status]


def _split_symbols(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.replace(",", " ").split() if item)


def _repo_path(resource_url: str, repo_root: Path) -> Path:
    parsed = urlparse(resource_url)
    relative = f"{parsed.netloc}{parsed.path}".lstrip("/")
    decoded = unquote(relative)
    candidate = repo_root / decoded
    if candidate.exists() or not decoded.startswith(LEGACY_SOURCE_PREFIX):
        return candidate
    stripped = decoded.removeprefix(LEGACY_SOURCE_PREFIX)
    legacy_candidate = repo_root.parent / "mss_agent" / stripped
    if legacy_candidate.exists():
        return legacy_candidate
    return repo_root / stripped


def _file_uri_path(resource_url: str) -> Path:
    parsed = urlparse(resource_url)
    return Path(unquote(parsed.path))
