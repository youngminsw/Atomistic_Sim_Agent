from __future__ import annotations

import json
from pathlib import Path

from sim_agent.materials import MaterialBuilderError, build_material_state
from sim_agent.schemas._parse import JsonMap, as_mapping, str_field
from sim_agent.schemas.errors import SchemaValidationError

from .lammps_contract import LAMMPSContractError, build_lammps_output_contract
from .logs import inspect_lammps_log


def validate_lammps_output_run(run_dir: Path, material_id: str, descriptor_root: Path) -> tuple[str, ...]:
    try:
        log_check = inspect_lammps_log(run_dir / "log.lammps")
    except FileNotFoundError:
        return ("missing:log.lammps",)
    if log_check.errors:
        return ("lammps_not_successful",)

    try:
        material = build_material_state(
            material_id=material_id,
            phases=("crystal",),
            descriptor_root=descriptor_root,
            method="fixture",
            pr_selectivity=20.0,
        )
        contract = build_lammps_output_contract("parsed-md-run", material.force_field)
    except (MaterialBuilderError, LAMMPSContractError) as exc:
        return (str(exc),)

    contract_report = contract.validate_output_dir(run_dir)
    if contract_report.error_lines:
        return contract_report.error_lines
    try:
        manifest = as_mapping(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")), "run_manifest")
        return _manifest_errors(manifest, material.force_field.protocol_id)
    except (json.JSONDecodeError, SchemaValidationError):
        return ("invalid_run_manifest",)


def _manifest_errors(manifest: JsonMap, force_field_protocol_id: str) -> tuple[str, ...]:
    errors: list[str] = []
    if str_field(manifest, "unit_style") != "metal":
        errors.append("unit_mismatch:unit_style")
    if str_field(manifest, "energy_unit") != "eV":
        errors.append("unit_mismatch:energy_unit")
    if str_field(manifest, "force_field_protocol_id") != force_field_protocol_id:
        errors.append("force_field_protocol_mismatch")
    return tuple(errors)
