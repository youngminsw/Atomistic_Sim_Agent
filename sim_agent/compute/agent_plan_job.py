from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from sim_agent.md_campaign import campaign_job_manifest_payload
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .source_payload import SOURCE_PAYLOAD_ARCHIVE
from .types import ComputePolicyError, JobBundleSpec

_LAMMPS_OUTPUT_FILENAMES = (
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
PRODUCTION_MD_INCIDENT_COUNT: Final = 500


def build_md_campaign_job_from_plan_dir(
    plan_dir: Path,
    environment_name: str,
    incident_count: int = PRODUCTION_MD_INCIDENT_COUNT,
) -> JobBundleSpec:
    if incident_count <= 0:
        raise ComputePolicyError("incident_count_must_be_positive")
    manifest = _load_mapping(plan_dir / "manifest.json", "manifest")
    campaign = _load_mapping(plan_dir / "md_campaign_plan.json", "md_campaign_plan")
    run_id = as_str(require(manifest, "run_id"), "run_id")
    job_id = f"{run_id}-md-campaign"
    output_path = f"artifacts/{job_id}/manifest.json"
    contract_output_path = f"artifacts/{job_id}/lammps_contract.json"
    schedule_output_path = f"artifacts/{job_id}/incident_schedule.json"
    surface_state_output_path = f"artifacts/{job_id}/surface_state.json"
    lammps_input_output_path = f"artifacts/{job_id}/in.atomistic_campaign"
    lammps_input_manifest_path = f"artifacts/{job_id}/lammps_input_manifest.json"
    structure_output_path = f"artifacts/{job_id}/surface_snapshot_before.data"
    potential_output_path = f"artifacts/{job_id}/Si.tersoff"
    assets_manifest_path = f"artifacts/{job_id}/lammps_assets_manifest.json"
    execution_plan_path = f"artifacts/{job_id}/lammps_execution_plan.json"
    return JobBundleSpec(
        job_id=job_id,
        environment_name=environment_name,
        command=_command(
            output_path,
            contract_output_path,
            schedule_output_path,
            surface_state_output_path,
            lammps_input_output_path,
            lammps_input_manifest_path,
            assets_manifest_path,
            execution_plan_path,
            incident_count,
        ),
        input_paths=(
            SOURCE_PAYLOAD_ARCHIVE,
            "manifest.json",
            "md_campaign_plan.json",
            "validated_request.json",
        ),
        output_paths=(
            output_path,
            contract_output_path,
            schedule_output_path,
            surface_state_output_path,
            lammps_input_output_path,
            lammps_input_manifest_path,
            structure_output_path,
            potential_output_path,
            assets_manifest_path,
            execution_plan_path,
        ),
        requires_cuda=False,
    )


def build_lammps_execution_job_from_md_campaign_job(
    md_campaign_job: JobBundleSpec,
) -> JobBundleSpec:
    artifact_dir = f"artifacts/{md_campaign_job.job_id}"
    plan_path = f"{artifact_dir}/lammps_execution_plan.json"
    result_path = f"{artifact_dir}/lammps_execution_result.json"
    return JobBundleSpec(
        job_id=_lammps_execution_job_id(md_campaign_job.job_id),
        environment_name=md_campaign_job.environment_name,
        command=(
            "python3",
            "02.Source_code/mss_agent/scripts/run_lammps_execution_plan.py",
            "--plan",
            plan_path,
            "--out",
            result_path,
            "--worker-capability",
            "worker_capability.json",
            "--execute",
        ),
        input_paths=(
            SOURCE_PAYLOAD_ARCHIVE,
            plan_path,
            f"{artifact_dir}/in.atomistic_campaign",
            f"{artifact_dir}/surface_snapshot_before.data",
            f"{artifact_dir}/Si.tersoff",
        ),
        output_paths=(result_path,) + _artifact_paths(artifact_dir, _LAMMPS_OUTPUT_FILENAMES),
        requires_cuda=False,
    )


def build_md_postprocess_job_from_lammps_execution_job(
    lammps_execution_job: JobBundleSpec,
    material_id: str,
) -> JobBundleSpec:
    artifact_dir = _artifact_dir_from_lammps_job(lammps_execution_job)
    events_path = f"{artifact_dir}/md_events.jsonl"
    report_path = f"{artifact_dir}/md_postprocess_report.json"
    return JobBundleSpec(
        job_id=_md_postprocess_job_id(lammps_execution_job.job_id),
        environment_name=lammps_execution_job.environment_name,
        command=(
            "python3",
            "02.Source_code/mss_agent/scripts/postprocess_lammps_execution.py",
            "--execution-result",
            f"{artifact_dir}/lammps_execution_result.json",
            "--material",
            material_id,
            "--descriptor-root",
            "02.Source_code/mss_agent/tests/fixtures/materials",
            "--events-out",
            events_path,
            "--report-out",
            report_path,
        ),
        input_paths=(SOURCE_PAYLOAD_ARCHIVE,) + lammps_execution_job.output_paths,
        output_paths=(events_path, report_path),
        requires_cuda=False,
    )


def _command(
    output_path: str,
    contract_output_path: str,
    schedule_output_path: str,
    surface_state_output_path: str,
    lammps_input_output_path: str,
    lammps_input_manifest_path: str,
    assets_manifest_path: str,
    execution_plan_path: str,
    incident_count: int,
) -> tuple[str, ...]:
    return (
        "python3",
        "02.Source_code/mss_agent/scripts/run_md_campaign_job.py",
        "--plan",
        "md_campaign_plan.json",
        "--request",
        "validated_request.json",
        "--descriptor-root",
        "02.Source_code/mss_agent/tests/fixtures/materials",
        "--out",
        output_path,
        "--contract-out",
        contract_output_path,
        "--incident-schedule-out",
        schedule_output_path,
        "--surface-state-out",
        surface_state_output_path,
        "--lammps-input-out",
        lammps_input_output_path,
        "--lammps-input-manifest-out",
        lammps_input_manifest_path,
        "--lammps-assets-manifest-out",
        assets_manifest_path,
        "--lammps-execution-plan-out",
        execution_plan_path,
        "--incident-count",
        str(incident_count),
    )


def _lammps_execution_job_id(md_campaign_job_id: str) -> str:
    suffix = "-md-campaign"
    if md_campaign_job_id.endswith(suffix):
        return f"{md_campaign_job_id.removesuffix(suffix)}-lammps-execution"
    return f"{md_campaign_job_id}-lammps-execution"


def _md_postprocess_job_id(lammps_execution_job_id: str) -> str:
    suffix = "-lammps-execution"
    if lammps_execution_job_id.endswith(suffix):
        return f"{lammps_execution_job_id.removesuffix(suffix)}-md-postprocess"
    return f"{lammps_execution_job_id}-md-postprocess"


def _artifact_paths(artifact_dir: str, filenames: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{artifact_dir}/{filename}" for filename in filenames)


def _artifact_dir_from_lammps_job(lammps_execution_job: JobBundleSpec) -> str:
    result_path = Path(lammps_execution_job.output_paths[0])
    return str(result_path.parent)


def _load_mapping(path: Path, field: str) -> JsonMap:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComputePolicyError(f"agent_plan_artifact_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"agent_plan_artifact_invalid_json={path}") from exc
    try:
        return as_mapping(payload, field)
    except SchemaValidationError as exc:
        raise ComputePolicyError(str(exc)) from exc
