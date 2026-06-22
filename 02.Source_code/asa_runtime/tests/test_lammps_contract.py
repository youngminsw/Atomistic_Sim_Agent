from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
MATERIAL_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "materials"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _force_field():
    from sim_agent.materials import build_material_state

    return build_material_state(
        material_id="Si",
        phases=("crystal",),
        descriptor_root=MATERIAL_ROOT,
        method="fixture",
        pr_selectivity=20.0,
    ).force_field


def test_lammps_contract_lists_required_parser_outputs_and_units() -> None:
    from sim_agent.md import build_lammps_output_contract

    contract = build_lammps_output_contract(run_id="ar-si-fixture", force_field=_force_field())

    assert contract.unit_system.unit_style == "metal"
    assert contract.unit_system.energy_unit == "eV"
    assert contract.unit_system.distance_unit == "angstrom"
    assert contract.unit_system.time_unit == "ps"
    assert contract.required_filenames == (
        "run_manifest.json",
        "surface_snapshot_before.data",
        "surface_snapshot_after.data",
        "incident.dump",
        "reflected.dump",
        "sputtered.dump",
        "implanted.dump",
        "traj.dump",
        "energy_depth_profile.csv",
        "damage_profile.csv",
        "roughness_rdf_descriptor.json",
        "log.lammps",
    )
    assert contract.collision_treatment.zbl_required is True
    assert contract.collision_treatment.force_field_protocol_id == "Si_Tersoff_ZBL_physical_v001"
    assert contract.manifest_payload()["unit_style"] == "metal"
    assert contract.manifest_payload()["energy_unit"] == "eV"


def test_lammps_contract_validation_reports_exact_missing_files(tmp_path: Path) -> None:
    from sim_agent.md import build_lammps_output_contract

    contract = build_lammps_output_contract(run_id="ar-si-fixture", force_field=_force_field())
    for filename in contract.required_filenames:
        if filename != "sputtered.dump":
            (tmp_path / filename).write_text("fixture\n", encoding="utf-8")

    report = contract.validate_output_dir(tmp_path)

    assert report.ok is False
    assert report.missing_filenames == ("sputtered.dump",)
    assert report.error_lines == ("missing:sputtered.dump",)


def test_lammps_contract_rejects_missing_force_field_provenance() -> None:
    from sim_agent.materials import ForceFieldRecord
    from sim_agent.md import LAMMPSContractError, build_lammps_output_contract

    force_field = ForceFieldRecord(
        material_id="Si",
        ion_species="Ar",
        protocol_id="",
        potential_name="Si.tersoff",
        source_url="",
        zbl_required=True,
    )

    try:
        build_lammps_output_contract(run_id="bad-force-field", force_field=force_field)
    except LAMMPSContractError as exc:
        assert str(exc) == "force_field_provenance_required"
    else:
        raise AssertionError("expected LAMMPSContractError")


def test_render_lammps_contract_cli_outputs_required_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "render_lammps_contract.py"),
            "--material",
            "Si",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--run-id",
            "cli-contract",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "required_outputs_ok=true" in result.stdout
    assert "units=metal" in result.stdout
    assert "energy_unit=eV" in result.stdout
    assert "zbl_required=true" in result.stdout
    assert "sputtered.dump" in result.stdout


def test_validate_md_output_contract_cli_reports_missing_sputtered_dump(tmp_path: Path) -> None:
    from sim_agent.md import build_lammps_output_contract

    contract = build_lammps_output_contract(run_id="ar-si-fixture", force_field=_force_field())
    for filename in contract.required_filenames:
        if filename != "sputtered.dump":
            (tmp_path / filename).write_text("fixture\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "validate_md_output_contract.py"),
            "--run-dir",
            str(tmp_path),
            "--material",
            "Si",
            "--descriptor-root",
            str(MATERIAL_ROOT),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "contract_valid=false" in result.stdout
    assert "missing:sputtered.dump" in result.stdout
