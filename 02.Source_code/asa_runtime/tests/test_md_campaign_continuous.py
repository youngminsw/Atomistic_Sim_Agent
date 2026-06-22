from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_plan_md_campaign_records_stratified_cumulative_bombardment() -> None:
    from sim_agent.md_campaign import plan_md_campaign

    plan = plan_md_campaign(
        material_id="Si",
        ion_species="Ar",
        phases=("crystal", "amorphous"),
        energy_range_eV=(20.0, 200.0),
        polar_range_deg=(0.0, 60.0),
        azimuth_range_deg=(0.0, 360.0),
    )

    assert plan.protocol_id == "continuous_stratified_bombardment"
    assert plan.event_probe_default is False
    assert plan.phases == ("crystal", "amorphous")
    assert plan.energy_strata.axis == "energy_eV"
    assert plan.energy_strata.minimum == 20.0
    assert plan.energy_strata.maximum == 200.0
    assert plan.polar_strata.axis == "polar_deg"
    assert plan.azimuth_strata.axis == "azimuth_deg"
    assert plan.pre_state_descriptors == (
        "amorphous_index",
        "damage_dose",
        "roughness_rms_nm",
        "rdf_order_features",
        "implanted_inert_fraction",
        "local_fluence",
        "removed_depth_nm",
    )
    assert plan.layer_renewal.removed_depth_threshold_nm == 1.0
    assert plan.layer_renewal.renewal_action == "expose_next_volume_state"


def test_plan_md_campaign_rejects_active_learning_for_new_material() -> None:
    from sim_agent.md_campaign import CampaignPlanError, ExpertReference, plan_md_campaign

    expert = ExpertReference(
        expert_id="Ar_on_Si",
        material_id="Si",
        ion_species="Ar",
        force_field_protocol_id="Si_Tersoff_ZBL_physical_v001",
        physics_scope="physical_bombardment_v1",
    )

    try:
        plan_md_campaign(
            material_id="PR",
            ion_species="Ar",
            phases=("amorphous",),
            energy_range_eV=(20.0, 200.0),
            polar_range_deg=(0.0, 60.0),
            azimuth_range_deg=(0.0, 360.0),
            extend_expert=expert,
        )
    except CampaignPlanError as exc:
        assert str(exc) == "new_campaign_required"
    else:
        raise AssertionError("expected CampaignPlanError")


def test_plan_md_campaign_cli_outputs_continuous_campaign() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_md_campaign.py"),
            "--material",
            "Si",
            "--ion",
            "Ar",
            "--phases",
            "crystal,amorphous",
            "--iedf",
            "20:200",
            "--iadf",
            "0:60",
            "--azimuth",
            "0:360",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "continuous_stratified_bombardment=true" in result.stdout
    assert "event_probe_default=false" in result.stdout
    assert "phases=crystal,amorphous" in result.stdout
    assert "energy_strata_eV=20.0:200.0" in result.stdout
    assert "layer_renewal=expose_next_volume_state@1.0nm" in result.stdout


def test_plan_md_campaign_cli_rejects_new_material_extension() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "plan_md_campaign.py"),
            "--extend-expert",
            "Ar_on_Si",
            "--material",
            "PR",
            "--ion",
            "Ar",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "md_campaign_plan_ok=false" in result.stdout
    assert "new_campaign_required" in result.stdout
