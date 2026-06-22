from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"
MATERIAL_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "materials"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_md_physics_gate_marks_small_campaign_as_smoke_only() -> None:
    from sim_agent.md import assess_md_physics_readiness
    from sim_agent.md_campaign import plan_md_campaign, md_campaign_plan_payload, stage_md_campaign

    # Given
    request = _load_request("valid_ar_si_pr_hole.json")
    campaign = md_campaign_plan_payload(
        plan_md_campaign(
            material_id="Si",
            ion_species="Ar",
            phases=("crystal", "amorphous"),
            energy_range_eV=(20.0, 200.0),
            polar_range_deg=(0.0, 60.0),
            azimuth_range_deg=(0.0, 360.0),
        )
    )
    staging = stage_md_campaign(campaign, request, MATERIAL_ROOT, incident_count=6)

    # When
    report = assess_md_physics_readiness(
        staging.manifest_payload,
        staging.lammps_contract_payload,
        staging.incident_schedule_payload,
        staging.surface_state_payload,
    )

    # Then
    assert report.ok is True
    assert report.production_ready is False
    assert report.payload["gate_status"] == "smoke_only"
    assert "zbl_collision_treatment_present" in report.payload["evidence"]
    assert "md_box_metadata_missing" in report.payload["blockers"]
    assert "production_incident_count_too_low:6<500" in report.payload["blockers"]


def test_stage_md_campaign_embeds_physics_gate_summary() -> None:
    from sim_agent.md_campaign import plan_md_campaign, md_campaign_plan_payload, stage_md_campaign

    # Given
    request = _load_request("valid_ar_si_pr_hole.json")
    campaign = md_campaign_plan_payload(
        plan_md_campaign(
            material_id="Si",
            ion_species="Ar",
            phases=("crystal", "amorphous"),
            energy_range_eV=(20.0, 200.0),
            polar_range_deg=(0.0, 60.0),
            azimuth_range_deg=(0.0, 360.0),
        )
    )

    # When
    staging = stage_md_campaign(campaign, request, MATERIAL_ROOT, incident_count=6)

    # Then
    assert staging.manifest_payload["physics_gate_status"] == "smoke_only"
    assert staging.manifest_payload["physics_production_ready"] is False
    assert "md_box_metadata_missing" in staging.manifest_payload["physics_blockers"]


def test_stage_md_campaign_can_reach_production_gate_with_md_box() -> None:
    from sim_agent.md_campaign import plan_md_campaign, md_campaign_plan_payload, stage_md_campaign

    request = _load_request("valid_ar_si_pr_hole.json")
    scene = as_mapping(request["scene"], "scene")
    surface = as_mapping(scene["surface_state"], "surface_state")
    surface["md_box"] = _production_md_box_payload()
    campaign = md_campaign_plan_payload(
        plan_md_campaign(
            material_id="Si",
            ion_species="Ar",
            phases=("crystal", "amorphous"),
            energy_range_eV=(20.0, 200.0),
            polar_range_deg=(0.0, 60.0),
            azimuth_range_deg=(0.0, 360.0),
        )
    )

    staging = stage_md_campaign(campaign, request, MATERIAL_ROOT, incident_count=500)

    assert staging.manifest_payload["physics_gate_status"] == "production_ready"
    assert staging.manifest_payload["physics_production_ready"] is True
    assert staging.surface_state_payload["md_box"]["atom_count"] == 5000


def test_validate_md_physics_gate_cli_writes_report(tmp_path: Path) -> None:
    from sim_agent.md_campaign import plan_md_campaign, md_campaign_plan_payload, stage_md_campaign

    # Given
    request = _load_request("valid_ar_si_pr_hole.json")
    campaign = md_campaign_plan_payload(
        plan_md_campaign(
            material_id="Si",
            ion_species="Ar",
            phases=("crystal", "amorphous"),
            energy_range_eV=(20.0, 200.0),
            polar_range_deg=(0.0, 60.0),
            azimuth_range_deg=(0.0, 360.0),
        )
    )
    staging = stage_md_campaign(campaign, request, MATERIAL_ROOT, incident_count=6)
    manifest_path = _write_json(tmp_path / "manifest.json", staging.manifest_payload)
    contract_path = _write_json(tmp_path / "lammps_contract.json", staging.lammps_contract_payload)
    schedule_path = _write_json(
        tmp_path / "incident_schedule.json",
        staging.incident_schedule_payload,
    )
    surface_path = _write_json(tmp_path / "surface_state.json", staging.surface_state_payload)
    report_path = tmp_path / "md_physics_gate.json"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "validate_md_physics_gate.py"),
            "--manifest",
            str(manifest_path),
            "--contract",
            str(contract_path),
            "--incident-schedule",
            str(schedule_path),
            "--surface-state",
            str(surface_path),
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
    assert "md_physics_gate_ok=true" in result.stdout
    assert "gate_status=smoke_only" in result.stdout
    assert payload["production_ready"] is False


def test_md_physics_gate_blocks_too_small_md_box_for_production() -> None:
    from sim_agent.md import assess_md_physics_readiness

    # Given
    report = assess_md_physics_readiness(
        _production_manifest_payload(),
        _production_contract_payload(),
        _production_schedule_payload(500),
        _surface_state_with_md_box(_too_small_md_box_payload()),
    )

    # Then
    assert report.ok is True
    assert report.production_ready is False
    assert report.payload["gate_status"] == "smoke_only"
    assert "md_box_lateral_size_too_small:x_nm=4.0<12.0" in report.payload["blockers"]
    assert "md_box_mobile_depth_too_shallow:5.0<9.0" in report.payload["blockers"]
    assert "md_box_atom_count_too_low:2000<3000" in report.payload["blockers"]


def test_md_physics_gate_accepts_production_sized_md_box() -> None:
    from sim_agent.md import assess_md_physics_readiness

    # Given
    report = assess_md_physics_readiness(
        _production_manifest_payload(),
        _production_contract_payload(),
        _production_schedule_payload(500),
        _surface_state_with_md_box(_production_md_box_payload()),
    )

    # Then
    assert report.ok is True
    assert report.production_ready is True
    assert report.payload["gate_status"] == "production_ready"
    assert "md_box_size_sufficient" in report.payload["evidence"]
    assert "md_box_regions_sufficient" in report.payload["evidence"]
    assert "md_box_timestep_run_length_sufficient" in report.payload["evidence"]
    md_box = as_mapping(report.payload["md_box"], "md_box")
    assert md_box["atom_count"] == 5000
    assert md_box["expected_cascade_depth_nm"] == 6.0


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def _write_json(path: Path, payload: JsonMap) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _production_manifest_payload() -> JsonMap:
    return {
        "material_id": "Si",
        "ion_species": "Ar",
    }


def _production_contract_payload() -> JsonMap:
    from sim_agent.md.lammps_contract import DEFAULT_OUTPUT_SPECS

    return {
        "unit_style": "metal",
        "energy_unit": "eV",
        "force_field_source_url": "repo://potentials/Si.tersoff",
        "force_field_protocol_id": "Si_Tersoff_ZBL_physical_v001",
        "zbl_required": True,
        "high_energy_collision_model": "zbl_overlay",
        "required_outputs": [spec.filename for spec in DEFAULT_OUTPUT_SPECS],
    }


def _production_schedule_payload(incident_count: int) -> JsonMap:
    return {
        "incident_count": incident_count,
        "events": [
            {
                "event_id": f"incident-{index + 1:06d}",
                "ion_species": "Ar",
                "energy_eV": 100.0,
                "polar_deg": 30.0,
                "azimuth_deg": 0.0,
            }
            for index in range(incident_count)
        ],
    }


def _surface_state_with_md_box(md_box: JsonMap) -> JsonMap:
    return {
        "descriptor_values": {
            "roughness_rms_nm": 0.1,
            "removed_depth_nm": 0.0,
        },
        "md_box": md_box,
    }


def _too_small_md_box_payload() -> JsonMap:
    return {
        "x_nm": 4.0,
        "y_nm": 8.0,
        "mobile_depth_nm": 5.0,
        "fixed_depth_nm": 0.5,
        "thermostat_depth_nm": 0.5,
        "expected_cascade_depth_nm": 6.0,
        "atom_count": 2000,
        "timestep_fs": 0.5,
        "run_length_ps": 0.5,
    }


def _production_md_box_payload() -> JsonMap:
    return {
        "x_nm": 16.0,
        "y_nm": 16.0,
        "mobile_depth_nm": 12.0,
        "fixed_depth_nm": 2.0,
        "thermostat_depth_nm": 2.0,
        "expected_cascade_depth_nm": 6.0,
        "atom_count": 5000,
        "timestep_fs": 0.1,
        "run_length_ps": 2.0,
    }
