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


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def _write_plan_artifacts(output_dir: Path) -> None:
    from sim_agent.agent_harness import (
        OfflineModelClient,
        SimulationAgentHarness,
        write_agent_plan_artifacts,
    )
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    endpoint = ModelProviderConfig.from_mapping(
        as_mapping(payload["llm_endpoint"], "llm_endpoint")
    )
    result = SimulationAgentHarness(
        endpoint=endpoint, client=OfflineModelClient()
    ).plan(payload)
    write_agent_plan_artifacts(output_dir, payload, result)


def test_lammps_input_deck_embeds_deterministic_incident_schedule() -> None:
    from sim_agent.md import render_lammps_input_deck

    deck = render_lammps_input_deck(
        _contract_payload(),
        _schedule_payload(),
        _surface_state_payload(),
    )

    assert deck.manifest_payload["schedule_id"] == "valid_ar_si_pr_hole-incident-schedule"
    assert deck.manifest_payload["surface_state_id"] == "valid_ar_si_pr_hole-surface-state"
    assert deck.manifest_payload["random_sampling_used"] is False
    assert deck.manifest_payload["required_structure_input"] == "surface_snapshot_before.data"
    assert "units metal" in deck.input_script
    assert "read_data surface_snapshot_before.data" in deck.input_script
    assert "# incident incident-000001" in deck.input_script
    assert "variable E_in equal 35.000000" in deck.input_script
    assert "variable polar_deg equal 5.000000" in deck.input_script
    assert "variable azimuth_deg equal 30.000000" in deck.input_script
    assert "dump dInc addatom custom 1 incident.dump" in deck.input_script
    assert "pair_style hybrid/overlay tersoff zbl 0.5 2.0" in deck.input_script
    assert "random(" not in deck.input_script


def test_lammps_run_assets_stage_required_structure_and_potential(tmp_path: Path) -> None:
    from sim_agent.md import stage_lammps_run_assets

    assets = stage_lammps_run_assets(
        _contract_payload(),
        _surface_state_payload(),
        tmp_path,
        PROJECT_ROOT,
    )

    structure_path = tmp_path / "surface_snapshot_before.data"
    potential_path = tmp_path / "Si.tersoff"
    assert assets.manifest_payload["assets_ready"] is True
    assert assets.manifest_payload["structure_filename"] == "surface_snapshot_before.data"
    assert assets.manifest_payload["potential_filename"] == "Si.tersoff"
    assert assets.manifest_payload["structure_source_kind"] == "repo_fixture"
    assert "1152 atoms" in structure_path.read_text(encoding="utf-8")
    assert "Tersoff parameters" in potential_path.read_text(encoding="utf-8")


def test_lammps_run_assets_require_explicit_relaxed_amorphous_structure(
    tmp_path: Path,
) -> None:
    from sim_agent.md import LAMMPSAssetStagingError, stage_lammps_run_assets

    try:
        stage_lammps_run_assets(
            _contract_payload(),
            _surface_state_payload() | {"phase": "amorphous"},
            tmp_path,
            PROJECT_ROOT,
        )
    except LAMMPSAssetStagingError as exc:
        assert str(exc) == "amorphous_lammps_structure_source_required"
    else:
        raise AssertionError("expected amorphous structure source gate")


def test_lammps_run_assets_stage_user_supplied_amorphous_structure(
    tmp_path: Path,
) -> None:
    from sim_agent.md import stage_lammps_run_assets

    source = tmp_path / "a_si_relaxed.data"
    source.write_text("relaxed amorphous Si fixture\n", encoding="utf-8")

    assets = stage_lammps_run_assets(
        _contract_payload(),
        _surface_state_payload()
        | {
            "phase": "amorphous",
            "lammps_structure_source": {
                "kind": "user_supplied",
                "path": source.as_uri(),
                "phase": "amorphous",
                "preparation": "melt_quench_relaxed_fixture",
            },
        },
        tmp_path / "run",
        PROJECT_ROOT,
    )

    assert assets.manifest_payload["phase"] == "amorphous"
    assert assets.manifest_payload["structure_source_kind"] == "user_supplied"
    assert "relaxed amorphous Si" in assets.structure_path.read_text(encoding="utf-8")


def test_lammps_run_assets_reject_structure_atom_count_mismatch(
    tmp_path: Path,
) -> None:
    from sim_agent.md import LAMMPSAssetStagingError, stage_lammps_run_assets

    source = tmp_path / "a_si_wrong_count.data"
    source.write_text("4999 atoms\n", encoding="utf-8")

    try:
        stage_lammps_run_assets(
            _contract_payload(),
            _surface_state_payload()
            | {
                "phase": "amorphous",
                "md_box": {"atom_count": 5000},
                "lammps_structure_source": {
                    "kind": "user_supplied",
                    "path": source.as_uri(),
                    "phase": "amorphous",
                    "preparation": "melt_quench_relaxed_fixture",
                },
            },
            tmp_path / "run",
            PROJECT_ROOT,
        )
    except LAMMPSAssetStagingError as exc:
        assert str(exc) == "lammps_structure_atom_count_mismatch:4999!=5000"
    else:
        raise AssertionError("expected atom-count mismatch gate")


def test_lammps_execution_plan_requires_staged_inputs(tmp_path: Path) -> None:
    from sim_agent.md import (
        build_lammps_execution_plan,
        render_lammps_input_deck,
        stage_lammps_run_assets,
    )

    deck = render_lammps_input_deck(
        _contract_payload(),
        _schedule_payload(),
        _surface_state_payload(),
    )
    stage_lammps_run_assets(
        _contract_payload(),
        _surface_state_payload(),
        tmp_path,
        PROJECT_ROOT,
    )
    (tmp_path / "in.atomistic_campaign").write_text(deck.input_script, encoding="utf-8")

    plan = build_lammps_execution_plan(
        deck.manifest_payload,
        _assets_manifest_payload(),
        tmp_path,
        lammps_binary="lmp",
    )

    assert plan.manifest_payload["execution_status"] == "ready_for_lammps"
    assert plan.manifest_payload["preflight_ok"] is True
    assert plan.manifest_payload["execute_now"] is False
    assert plan.manifest_payload["working_directory"] == str(tmp_path)
    assert plan.manifest_payload["input_deck"] == "in.atomistic_campaign"
    assert plan.manifest_payload["expected_incident_count"] == 2
    assert plan.manifest_payload["command_line"] == (
        f"cd {tmp_path} && lmp -in in.atomistic_campaign"
    )
    assert "surface_snapshot_before.data" in plan.manifest_payload["required_inputs"]
    assert "log.lammps" in plan.manifest_payload["expected_outputs"]


def test_run_md_campaign_job_writes_lammps_input_deck(tmp_path: Path) -> None:
    _write_plan_artifacts(tmp_path)
    out_path = tmp_path / "md_campaign_job_manifest.json"
    contract_path = tmp_path / "lammps_contract.json"
    schedule_path = tmp_path / "incident_schedule.json"
    surface_state_path = tmp_path / "surface_state.json"
    input_path = tmp_path / "in.atomistic_campaign"
    input_manifest_path = tmp_path / "lammps_input_manifest.json"
    assets_manifest_path = tmp_path / "lammps_assets_manifest.json"
    execution_plan_path = tmp_path / "lammps_execution_plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_md_campaign_job.py"),
            "--plan",
            str(tmp_path / "md_campaign_plan.json"),
            "--request",
            str(tmp_path / "validated_request.json"),
            "--descriptor-root",
            str(MATERIAL_ROOT),
            "--out",
            str(out_path),
            "--contract-out",
            str(contract_path),
            "--incident-schedule-out",
            str(schedule_path),
            "--surface-state-out",
            str(surface_state_path),
            "--lammps-input-out",
            str(input_path),
            "--lammps-input-manifest-out",
            str(input_manifest_path),
            "--lammps-assets-manifest-out",
            str(assets_manifest_path),
            "--lammps-execution-plan-out",
            str(execution_plan_path),
            "--incident-count",
            "6",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = as_mapping(
        json.loads(input_manifest_path.read_text(encoding="utf-8")),
        "lammps_input_manifest",
    )
    assets_manifest = as_mapping(
        json.loads(assets_manifest_path.read_text(encoding="utf-8")),
        "lammps_assets_manifest",
    )
    execution_plan = as_mapping(
        json.loads(execution_plan_path.read_text(encoding="utf-8")),
        "lammps_execution_plan",
    )
    input_script = input_path.read_text(encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lammps_input_path=" in result.stdout
    assert "lammps_assets_manifest_path=" in result.stdout
    assert "lammps_execution_plan_path=" in result.stdout
    assert manifest["incident_count"] == 6
    assert manifest["force_field_protocol_id"] == "Si_Tersoff_ZBL_physical_v001"
    assert assets_manifest["assets_ready"] is True
    assert execution_plan["execution_status"] == "ready_for_lammps"
    assert execution_plan["command_line"].endswith("lmp -in in.atomistic_campaign")
    assert (tmp_path / "surface_snapshot_before.data").exists()
    assert (tmp_path / "Si.tersoff").exists()
    assert "incident-000006" in input_script
    assert "variable E_in equal 185.000000" in input_script
    assert "random(" not in input_script


def _contract_payload() -> JsonMap:
    return {
        "run_id": "valid_ar_si_pr_hole-lammps-contract",
        "unit_style": "metal",
        "distance_unit": "angstrom",
        "time_unit": "ps",
        "energy_unit": "eV",
        "timestep_unit": "ps",
        "required_outputs": [
            "run_manifest.json",
            "incident.dump",
            "reflected.dump",
            "sputtered.dump",
            "implanted.dump",
            "traj.dump",
            "log.lammps",
        ],
        "zbl_required": True,
        "high_energy_collision_model": "zbl_overlay",
        "force_field_protocol_id": "Si_Tersoff_ZBL_physical_v001",
        "force_field_source_url": (
            "repo://02.Source_code/mss_agent/md_agent_window/Reference/"
            "force_field_library/potentials/Si.tersoff"
        ),
    }


def _schedule_payload() -> JsonMap:
    return {
        "schedule_id": "valid_ar_si_pr_hole-incident-schedule",
        "sampling_policy": "deterministic_stratified_midpoint_v1",
        "incident_count": 2,
        "ion_species": "Ar",
        "events": [
            {
                "event_id": "incident-000001",
                "ion_species": "Ar",
                "energy_eV": 35.0,
                "polar_deg": 5.0,
                "azimuth_deg": 30.0,
            },
            {
                "event_id": "incident-000002",
                "ion_species": "Ar",
                "energy_eV": 65.0,
                "polar_deg": 15.0,
                "azimuth_deg": 90.0,
            },
        ],
    }


def _surface_state_payload() -> JsonMap:
    return {
        "surface_state_id": "valid_ar_si_pr_hole-surface-state",
        "material_id": "Si",
        "phase": "crystal",
        "descriptor_values": {
            "roughness_rms_nm": 0.1,
            "removed_depth_nm": 0.0,
            "damage_dose": 0.0,
        },
    }


def _assets_manifest_payload() -> JsonMap:
    return {
        "asset_manifest_id": "valid_ar_si_pr_hole-lammps-contract-assets",
        "run_id": "valid_ar_si_pr_hole-lammps-contract",
        "surface_state_id": "valid_ar_si_pr_hole-surface-state",
        "assets_ready": True,
        "material_id": "Si",
        "phase": "crystal",
        "structure_filename": "surface_snapshot_before.data",
        "potential_filename": "Si.tersoff",
        "force_field_protocol_id": "Si_Tersoff_ZBL_physical_v001",
    }
