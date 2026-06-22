from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, float_map, optional_str, str_field

from .types import ForceFieldRecord, MaterialBuildReport, MaterialBuilderError, MaterialDescriptor, PRMaterial


def build_material_state(
    material_id: str,
    phases: tuple[str, ...],
    descriptor_root: Path,
    method: str,
    pr_selectivity: float,
) -> MaterialBuildReport:
    force_field = _force_field_for(material_id)
    pr_material = build_pr_material("PR", pr_selectivity)
    crystal = _load_phase_descriptor(material_id, "crystal", descriptor_root) if "crystal" in phases else None
    amorphous = _amorphous_descriptor(material_id, descriptor_root, method) if "amorphous" in phases else None
    return MaterialBuildReport(
        material_id=material_id,
        crystal=crystal,
        amorphous=amorphous,
        pr_material=pr_material,
        force_field=force_field,
        dry_run=True,
    )


def build_pr_material(material_id: str, selectivity: float) -> PRMaterial:
    if selectivity <= 0.0:
        raise MaterialBuilderError("pr_selectivity_must_be_positive")
    return PRMaterial(
        material_id=material_id,
        role="mask",
        phase="amorphous",
        selectivity=selectivity,
        relative_erosion_rate=1.0 / selectivity,
    )


def material_report_payload(report: MaterialBuildReport) -> JsonMap:
    return {
        "material_id": report.material_id,
        "dry_run": report.dry_run,
        "crystal": None if report.crystal is None else _descriptor_payload(report.crystal),
        "amorphous": None if report.amorphous is None else _descriptor_payload(report.amorphous),
        "pr_material": {
            "material_id": report.pr_material.material_id,
            "role": report.pr_material.role,
            "phase": report.pr_material.phase,
            "selectivity": report.pr_material.selectivity,
            "relative_erosion_rate": report.pr_material.relative_erosion_rate,
        },
        "force_field": {
            "material_id": report.force_field.material_id,
            "ion_species": report.force_field.ion_species,
            "protocol_id": report.force_field.protocol_id,
            "potential_name": report.force_field.potential_name,
            "source_url": report.force_field.source_url,
            "zbl_required": report.force_field.zbl_required,
        },
    }


def _amorphous_descriptor(material_id: str, descriptor_root: Path, method: str) -> MaterialDescriptor:
    if method == "random_disorder":
        raise MaterialBuilderError("relaxation_required")
    return _load_phase_descriptor(material_id, "amorphous", descriptor_root)


def _load_phase_descriptor(material_id: str, phase: str, descriptor_root: Path) -> MaterialDescriptor:
    path = descriptor_root / f"{material_id.lower()}_{phase}_descriptor.json"
    try:
        payload = as_mapping(json.loads(path.read_text(encoding="utf-8")), "descriptor")
    except FileNotFoundError as exc:
        raise MaterialBuilderError(f"{phase}_descriptor_required") from exc
    except json.JSONDecodeError as exc:
        raise MaterialBuilderError("invalid_descriptor_json") from exc
    descriptor = _descriptor_from_mapping(payload)
    if descriptor.phase != phase or descriptor.material_id != material_id:
        raise MaterialBuilderError("descriptor_identity_mismatch")
    if phase == "amorphous" and not _relaxed_amorphous(descriptor):
        raise MaterialBuilderError("relaxation_required")
    return descriptor


def _descriptor_from_mapping(value: JsonMap) -> MaterialDescriptor:
    return MaterialDescriptor(
        structure_id=str_field(value, "structure_id"),
        material_id=str_field(value, "material_id"),
        phase=str_field(value, "phase"),
        density_g_cm3=as_float(value.get("density_g_cm3"), "density_g_cm3"),
        rdf_order_features=float_map(value.get("rdf_order_features", {}), "rdf_order_features"),
        orientation=optional_str(value, "orientation") or "",
        preparation=optional_str(value, "preparation") or "",
    )


def _descriptor_payload(descriptor: MaterialDescriptor) -> JsonMap:
    return {
        "structure_id": descriptor.structure_id,
        "material_id": descriptor.material_id,
        "phase": descriptor.phase,
        "density_g_cm3": descriptor.density_g_cm3,
        "rdf_order_features": descriptor.rdf_order_features,
        "orientation": descriptor.orientation,
        "preparation": descriptor.preparation,
    }


def _relaxed_amorphous(descriptor: MaterialDescriptor) -> bool:
    preparation = descriptor.preparation.lower()
    return "relax" in preparation or "import" in preparation


def _force_field_for(material_id: str) -> ForceFieldRecord:
    if material_id != "Si":
        raise MaterialBuilderError("force_field_provenance_required")
    return ForceFieldRecord(
        material_id="Si",
        ion_species="Ar",
        protocol_id="Si_Tersoff_ZBL_physical_v001",
        potential_name="Si.tersoff + ZBL overlay",
        source_url="repo://02.Source_code/mss_agent/md_agent_window/Reference/force_field_library/potentials/Si.tersoff",
        zbl_required=True,
    )
