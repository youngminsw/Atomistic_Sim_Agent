from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _fixture(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_2d_image_request_keeps_path_description_and_asks_for_distributions() -> None:
    from sim_agent.input_planner import plan_simulation_input

    result = plan_simulation_input(_fixture("2d_image_missing_distribution.json"))

    assert result.mode == "2d"
    assert result.feature_type == "trench"
    assert result.geometry_kind == "image"
    assert result.geometry_path == "tests/fixtures/geometry/pr_trench.png"
    assert result.structure_description == "2D PR trench image mask with exposed silicon opening"
    assert result.clarification_required is True
    assert result.missing_fields == ("iedf", "iadf")
    assert "IonEnergyDistribution" in result.clarifications[0].question
    assert result.model_training_required is False


def test_3d_mesh_request_records_materials_units_and_available_expert() -> None:
    from sim_agent.input_planner import plan_simulation_input

    result = plan_simulation_input(_fixture("valid_ar_si_pr_hole.json"))

    assert result.mode == "3d"
    assert result.feature_type == "hole"
    assert result.geometry_kind == "mesh"
    assert result.geometry_units == "nm"
    assert result.mask_material_id == "PR"
    assert result.target_material_id == "Si"
    assert result.trained_kernel_id == "Ar_on_Si__physical_v001"
    assert result.model_training_required is False
    assert result.clarification_required is False


def test_missing_iedf_does_not_invent_fixed_energy() -> None:
    from sim_agent.input_planner import plan_simulation_input

    result = plan_simulation_input(_fixture("missing_iedf.json"))

    assert result.clarification_required is True
    assert result.missing_fields == ("iedf",)
    assert result.proposed_defaults == ()


def test_unknown_material_requires_model_training() -> None:
    from sim_agent.input_planner import plan_simulation_input

    result = plan_simulation_input(_fixture("ar_on_unknown_material.json"))

    assert result.ion_species == "Ar"
    assert result.target_material_id == "UnobtaniumFixture"
    assert result.model_training_required is True
    assert result.training_reason == "no_trained_expert_for_Ar_on_UnobtaniumFixture"


def test_plan_request_cli_reports_2d_clarification() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_request.py"),
            "--fixture",
            str(FIXTURE_ROOT / "2d_image_missing_distribution.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mode=2d" in result.stdout
    assert "clarification_required=true" in result.stdout
    assert "missing_fields=iedf,iadf" in result.stdout


def test_plan_request_cli_reports_training_required() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_request.py"),
            "--fixture",
            str(FIXTURE_ROOT / "ar_on_unknown_material.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_training_required=true" in result.stdout
    assert "training_reason=no_trained_expert_for_Ar_on_UnobtaniumFixture" in result.stdout
