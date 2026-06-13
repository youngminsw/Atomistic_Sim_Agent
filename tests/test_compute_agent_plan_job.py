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

def test_build_md_campaign_job_from_agent_plan_artifacts(tmp_path: Path) -> None:
    from sim_agent.compute import build_md_campaign_job_from_plan_dir, build_worker_bundle

    _write_plan_artifacts(tmp_path)

    job = build_md_campaign_job_from_plan_dir(tmp_path, environment_name="atomistic-sim-gpu")
    bundle = build_worker_bundle("gpu-5090", job, remote_user="swym")

    assert job.job_id == "plan-valid_ar_si_pr_hole-md-campaign"
    assert job.command == (
        "python3",
        "02.Source_code/mss_agent/scripts/run_md_campaign_job.py",
        "--plan",
        "md_campaign_plan.json",
        "--request",
        "validated_request.json",
        "--descriptor-root",
        "02.Source_code/mss_agent/tests/fixtures/materials",
        "--out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/manifest.json",
        "--contract-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_contract.json",
        "--incident-schedule-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/incident_schedule.json",
        "--surface-state-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_state.json",
        "--lammps-input-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/in.atomistic_campaign",
        "--lammps-input-manifest-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_input_manifest.json",
        "--lammps-assets-manifest-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_assets_manifest.json",
        "--lammps-execution-plan-out",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_execution_plan.json",
        "--incident-count",
        "500",
    )
    assert job.input_paths == (
        "source_payload.tar.gz",
        "manifest.json",
        "md_campaign_plan.json",
        "validated_request.json",
    )
    expected_outputs = (
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_contract.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/incident_schedule.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_state.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/in.atomistic_campaign",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_input_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_snapshot_before.data",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/Si.tersoff",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_assets_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_execution_plan.json",
    )
    assert job.output_paths == expected_outputs
    assert job.requires_cuda is False
    assert "upload:md_campaign_plan.json->" in bundle.transfer_plan[2]
    assert "nvidia-smi" not in bundle.preflight_commands

def test_run_md_campaign_job_writes_dry_run_manifest(tmp_path: Path) -> None:
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

    manifest = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "manifest")
    contract = as_mapping(json.loads(contract_path.read_text(encoding="utf-8")), "contract")
    schedule = as_mapping(json.loads(schedule_path.read_text(encoding="utf-8")), "schedule")
    surface = as_mapping(json.loads(surface_state_path.read_text(encoding="utf-8")), "surface")
    input_manifest = as_mapping(
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
    assert "md_campaign_job_ok=true" in result.stdout
    assert manifest["run_status"] == "dry_run_ready"
    assert manifest["request_id"] == "valid_ar_si_pr_hole"
    assert manifest["material_id"] == "Si"
    assert manifest["ion_species"] == "Ar"
    assert manifest["preflight_gate"] == "lammps_contract_ready"
    assert manifest["force_field_protocol_id"] == "Si_Tersoff_ZBL_physical_v001"
    assert manifest["zbl_required"] is True
    assert manifest["unit_style"] == "metal"
    assert "sputtered.dump" in manifest["required_outputs"]
    assert contract["run_id"] == "valid_ar_si_pr_hole-lammps-contract"
    assert contract["force_field_protocol_id"] == "Si_Tersoff_ZBL_physical_v001"
    assert "roughness_rdf_descriptor.json" in contract["required_outputs"]
    assert schedule["schedule_id"] == "valid_ar_si_pr_hole-incident-schedule"
    assert schedule["incident_count"] == 6
    assert schedule["sampling_policy"] == "deterministic_stratified_midpoint_v1"
    events = schedule["events"]
    assert isinstance(events, list)
    assert events[0]["event_id"] == "incident-000001"
    assert events[0]["energy_eV"] == 35.0
    assert events[0]["polar_deg"] == 5.0
    assert events[0]["azimuth_deg"] == 30.0
    assert surface["surface_state_id"] == "valid_ar_si_pr_hole-surface-state"
    assert surface["layer_renewal_action"] == "expose_next_volume_state"
    assert surface["removed_depth_threshold_nm"] == 1.0
    descriptor_values = as_mapping(surface["descriptor_values"], "descriptor_values")
    assert descriptor_values["roughness_rms_nm"] == 0.1
    assert descriptor_values["rdf_crystal_similarity"] == 0.98
    assert input_manifest["schedule_id"] == "valid_ar_si_pr_hole-incident-schedule"
    assert input_manifest["random_sampling_used"] is False
    assert assets_manifest["assets_ready"] is True
    assert execution_plan["execution_status"] == "ready_for_lammps"
    assert execution_plan["command_line"].endswith("lmp -in in.atomistic_campaign")
    assert (tmp_path / "surface_snapshot_before.data").exists()
    assert (tmp_path / "Si.tersoff").exists()
    assert "incident-000006" in input_script
    assert "random(" not in input_script


def test_prepare_md_campaign_worker_bundle_cli_writes_job_and_worker_json(tmp_path: Path) -> None:
    _write_plan_artifacts(tmp_path)
    job_path = tmp_path / "md_campaign_job.json"
    worker_path = tmp_path / "worker_bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "prepare_md_campaign_worker_bundle.py"),
            "--plan-dir",
            str(tmp_path),
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--remote-user",
            "swym",
            "--job-out",
            str(job_path),
            "--worker-out",
            str(worker_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    job = as_mapping(json.loads(job_path.read_text(encoding="utf-8")), "job")
    worker = as_mapping(json.loads(worker_path.read_text(encoding="utf-8")), "worker")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "md_campaign_worker_bundle_ok=true" in result.stdout
    assert job["job_id"] == "plan-valid_ar_si_pr_hole-md-campaign"
    assert job["outputs"] == [
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_contract.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/incident_schedule.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_state.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/in.atomistic_campaign",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_input_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_snapshot_before.data",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/Si.tersoff",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_assets_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_execution_plan.json",
    ]
    assert worker["host_alias"] == "gpu-5090"
    assert worker["environment_name"] == "atomistic-sim-gpu"
    assert worker["output_paths"] == job["outputs"]
