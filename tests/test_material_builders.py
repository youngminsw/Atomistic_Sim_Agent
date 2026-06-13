from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
MATERIAL_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "materials"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_si_crystal_and_amorphous_descriptors_validate() -> None:
    from sim_agent.materials import build_material_state

    report = build_material_state(
        material_id="Si",
        phases=("crystal", "amorphous"),
        descriptor_root=MATERIAL_ROOT,
        method="fixture",
        pr_selectivity=20.0,
    )

    assert report.crystal is not None
    assert report.crystal.structure_id == "si_100_relaxed"
    assert report.crystal.orientation == "100"
    assert report.crystal.rdf_order_features["crystal_similarity"] == 0.98
    assert report.amorphous is not None
    assert report.amorphous.structure_id == "a_si_melt_quench_relaxed"
    assert report.amorphous.preparation == "melt_quench_relax_fixture"
    assert report.amorphous.rdf_order_features["amorphous_similarity"] == 0.88
    assert report.force_field.protocol_id == "Si_Tersoff_ZBL_physical_v001"


def test_random_amorphous_without_relaxation_is_rejected() -> None:
    from sim_agent.materials import MaterialBuilderError, build_material_state

    try:
        build_material_state(
            material_id="Si",
            phases=("amorphous",),
            descriptor_root=MATERIAL_ROOT,
            method="random_disorder",
            pr_selectivity=20.0,
        )
    except MaterialBuilderError as exc:
        assert str(exc) == "relaxation_required"
    else:
        raise AssertionError("expected MaterialBuilderError")


def test_pr_material_selectivity_is_part_of_material_model() -> None:
    from sim_agent.materials import build_pr_material

    pr = build_pr_material(material_id="PR", selectivity=35.0)

    assert pr.material_id == "PR"
    assert pr.phase == "amorphous"
    assert pr.role == "mask"
    assert pr.selectivity == 35.0
    assert pr.relative_erosion_rate == 1.0 / 35.0


def test_missing_force_field_provenance_blocks_md_campaign_planning() -> None:
    from sim_agent.materials import MaterialBuilderError, build_material_state

    try:
        build_material_state(
            material_id="UnobtaniumFixture",
            phases=("crystal",),
            descriptor_root=MATERIAL_ROOT,
            method="fixture",
            pr_selectivity=20.0,
        )
    except MaterialBuilderError as exc:
        assert str(exc) == "force_field_provenance_required"
    else:
        raise AssertionError("expected MaterialBuilderError")


def test_build_material_state_cli_validates_crystal_and_amorphous(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_material_state.py"),
            "--material",
            "Si",
            "--phases",
            "crystal,amorphous",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--out",
            str(tmp_path / "material_report.json"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "crystal_valid=true" in result.stdout
    assert "amorphous_valid=true" in result.stdout
    assert "descriptor_written=true" in result.stdout
    assert (tmp_path / "material_report.json").exists()


def test_build_material_state_cli_rejects_random_amorphous() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_material_state.py"),
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--method",
            "random_disorder",
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "relaxation_required" in result.stdout
