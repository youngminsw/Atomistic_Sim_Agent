from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_bool, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


REAXFF_PAIR_STYLES: Final = ("reaxff", "reax/c")
SOURCE_PREFIXES: Final = ("repo://", "https://", "http://", "file://")


@dataclass(frozen=True, slots=True)
class PotentialValidationReport:
    ok: bool
    payload: JsonMap


def validate_potential_candidate(
    candidate: JsonMap,
    material_id: str,
    ion_species: str,
    required_elements: tuple[str, ...],
) -> PotentialValidationReport:
    errors: list[str] = []
    evidence: list[str] = []
    potential_id = _required_text(candidate, "potential_id", errors)
    candidate_material = _required_text(candidate, "material_id", errors)
    candidate_ion = _required_text(candidate, "ion_species", errors)
    pair_style = _required_text(candidate, "pair_style", errors).lower()
    lammps_unit_style = _required_text(candidate, "lammps_unit_style", errors).lower()
    source_url = _required_text(candidate, "source_url", errors)
    element_symbols = _required_text_tuple(candidate, "element_symbols", errors)
    atom_type_mapping = _required_text_tuple(candidate, "atom_type_mapping", errors)

    _record_identity_errors(candidate_material, material_id, candidate_ion, ion_species, errors)
    _record_source_evidence(source_url, errors, evidence)
    _record_provenance(candidate, errors, evidence)
    _record_element_mapping(
        required_elements,
        element_symbols,
        atom_type_mapping,
        errors,
        evidence,
    )
    _record_required_text(candidate, "potential_name", errors, evidence)
    _record_required_text(candidate, "license", errors, evidence)
    _record_required_text(candidate, "fitted_system", errors, evidence)
    _record_required_text(candidate, "transferability_scope", errors, evidence)
    _record_syntax_smoke(candidate, errors, evidence)
    _record_reaxff_rules(pair_style, lammps_unit_style, candidate, errors, evidence)

    ok = not errors
    gate_status = "potential_candidate_accepted" if ok else "potential_candidate_rejected"
    return PotentialValidationReport(
        ok=ok,
        payload={
            "ok": ok,
            "gate_status": gate_status,
            "potential_id": potential_id,
            "material_id": candidate_material,
            "ion_species": candidate_ion,
            "pair_style": pair_style,
            "lammps_unit_style": lammps_unit_style,
            "element_symbols": list(element_symbols),
            "atom_type_mapping": list(atom_type_mapping),
            "evidence": evidence,
            "errors": errors,
        },
    )


def _required_text(candidate: JsonMap, field: str, errors: list[str]) -> str:
    try:
        return as_str(require(candidate, field), field)
    except SchemaValidationError:
        errors.append(f"{field}_required")
        return ""


def _required_text_tuple(candidate: JsonMap, field: str, errors: list[str]) -> tuple[str, ...]:
    try:
        values = as_sequence(require(candidate, field), field)
    except SchemaValidationError:
        errors.append(f"{field}_required")
        return ()
    return _text_tuple(values, field, errors)


def _text_tuple(values: Sequence[object], field: str, errors: list[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for index, value in enumerate(values):
        try:
            parsed.append(as_str(value, f"{field}[{index}]"))
        except SchemaValidationError:
            errors.append(f"{field}_invalid")
    return tuple(parsed)


def _record_identity_errors(
    candidate_material: str,
    material_id: str,
    candidate_ion: str,
    ion_species: str,
    errors: list[str],
) -> None:
    if candidate_material and candidate_material != material_id:
        errors.append("material_id_mismatch")
    if candidate_ion and candidate_ion != ion_species:
        errors.append("ion_species_mismatch")


def _record_source_evidence(source_url: str, errors: list[str], evidence: list[str]) -> None:
    if not source_url:
        return
    if source_url.startswith(SOURCE_PREFIXES):
        evidence.append("source_url_present")
        return
    errors.append("source_url_scheme_unsupported")


def _record_provenance(candidate: JsonMap, errors: list[str], evidence: list[str]) -> None:
    provenance = _first_present_text(
        candidate,
        ("provenance_url", "publication_url", "publication_doi"),
    )
    if provenance:
        evidence.append("provenance_present")
        return
    errors.append("provenance_required")


def _record_element_mapping(
    required_elements: tuple[str, ...],
    element_symbols: tuple[str, ...],
    atom_type_mapping: tuple[str, ...],
    errors: list[str],
    evidence: list[str],
) -> None:
    missing_symbols = tuple(
        element for element in required_elements if element not in element_symbols
    )
    missing_mapping = tuple(
        element for element in required_elements if element not in atom_type_mapping
    )
    if missing_symbols:
        errors.append("element_symbols_incomplete")
    if missing_mapping:
        errors.append("atom_type_mapping_incomplete")
    if not missing_symbols and not missing_mapping and required_elements:
        evidence.append("element_mapping_complete")


def _record_required_text(
    candidate: JsonMap,
    field: str,
    errors: list[str],
    evidence: list[str],
) -> None:
    value = _first_present_text(candidate, (field,))
    if value:
        evidence.append(f"{field}_present")
        return
    errors.append(f"{field}_required")


def _record_syntax_smoke(candidate: JsonMap, errors: list[str], evidence: list[str]) -> None:
    try:
        smoke_passed = as_bool(require(candidate, "syntax_smoke_passed"), "syntax_smoke_passed")
    except SchemaValidationError:
        errors.append("syntax_smoke_required")
        return
    if smoke_passed:
        evidence.append("syntax_smoke_passed")
        return
    errors.append("syntax_smoke_failed")


def _record_reaxff_rules(
    pair_style: str,
    lammps_unit_style: str,
    candidate: JsonMap,
    errors: list[str],
    evidence: list[str],
) -> None:
    if pair_style not in REAXFF_PAIR_STYLES:
        return
    if lammps_unit_style != "real":
        errors.append("reaxff_real_units_required")
    publication = _first_present_text(candidate, ("publication_url", "publication_doi"))
    if publication:
        evidence.append("reaxff_publication_present")
    else:
        errors.append("reaxff_publication_required")
    if _first_present_text(candidate, ("fitted_system",)):
        evidence.append("reaxff_fitted_system_present")
    else:
        errors.append("reaxff_fitted_system_required")


def _first_present_text(candidate: JsonMap, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = candidate.get(field)
        if isinstance(value, str) and value:
            return value
    return ""
