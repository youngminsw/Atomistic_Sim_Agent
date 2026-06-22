from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_validate_potential_candidate_accepts_provenanced_zbl_overlay() -> None:
    from sim_agent.materials import validate_potential_candidate

    # Given
    candidate = _tersiff_zbl_candidate()

    # When
    report = validate_potential_candidate(
        candidate,
        material_id="Si",
        ion_species="Ar",
        required_elements=("Si", "Ar"),
    )

    # Then
    assert report.ok is True
    assert report.payload["gate_status"] == "potential_candidate_accepted"
    assert report.payload["potential_id"] == "si-tersoff-zbl-v001"
    assert "syntax_smoke_passed" in report.payload["evidence"]


def test_validate_potential_candidate_rejects_unprovenanced_reaxff() -> None:
    from sim_agent.materials import validate_potential_candidate

    # Given
    candidate = _reaxff_candidate_without_publication()

    # When
    report = validate_potential_candidate(
        candidate,
        material_id="SiO2",
        ion_species="Ar",
        required_elements=("Si", "O", "Ar"),
    )

    # Then
    assert report.ok is False
    assert report.payload["gate_status"] == "potential_candidate_rejected"
    assert "reaxff_real_units_required" in report.payload["errors"]
    assert "reaxff_publication_required" in report.payload["errors"]
    assert "reaxff_fitted_system_required" in report.payload["errors"]


def test_validate_potential_candidate_cli_writes_report(tmp_path: Path) -> None:
    # Given
    candidate_path = tmp_path / "potential_candidate.json"
    report_path = tmp_path / "potential_report.json"
    candidate_path.write_text(json.dumps(_tersiff_zbl_candidate()), encoding="utf-8")

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "validate_potential_candidate.py"),
            "--candidate",
            str(candidate_path),
            "--material",
            "Si",
            "--ion",
            "Ar",
            "--required-elements",
            "Si,Ar",
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(report_path.read_text(encoding="utf-8")), "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "potential_gate_ok=true" in result.stdout
    assert payload["gate_status"] == "potential_candidate_accepted"


def _tersiff_zbl_candidate() -> JsonMap:
    return {
        "potential_id": "si-tersoff-zbl-v001",
        "material_id": "Si",
        "ion_species": "Ar",
        "pair_style": "hybrid/overlay tersoff zbl",
        "potential_name": "Si.tersoff + ZBL overlay",
        "source_url": "repo://md_agent_window/Reference/force_field_library/potentials/Si.tersoff",
        "provenance_url": "repo://tests/fixtures/materials/si_crystal_descriptor.json",
        "license": "repo-local",
        "lammps_unit_style": "metal",
        "element_symbols": ["Si", "Ar"],
        "atom_type_mapping": ["Si", "Ar"],
        "syntax_smoke_passed": True,
        "fitted_system": "Si collision cascade with ZBL overlay for Ar projectile",
        "transferability_scope": (
            "Ar physical sputtering of Si with explicit ZBL high-energy treatment"
        ),
    }


def _reaxff_candidate_without_publication() -> JsonMap:
    return {
        "potential_id": "bad-reaxff",
        "material_id": "SiO2",
        "ion_species": "Ar",
        "pair_style": "reaxff",
        "potential_name": "unvetted ffield.reax",
        "source_url": "https://example.invalid/ffield.reax",
        "license": "unknown",
        "lammps_unit_style": "metal",
        "element_symbols": ["Si", "O", "Ar"],
        "atom_type_mapping": ["Si", "O", "Ar"],
        "syntax_smoke_passed": True,
        "transferability_scope": "element match only",
    }
