from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.materials import MaterialBuilderError, build_material_state
from sim_agent.md import LAMMPSContractError, build_lammps_output_contract
from sim_agent.md.physics_gate import MDPhysicsReadinessReport, assess_md_physics_readiness
from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, as_sequence, as_str, require

from .inputs import MDCampaignInputError, incident_schedule_payload, surface_state_payload


@dataclass(frozen=True, slots=True)
class MDCampaignStagingError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class MDCampaignStagingBundle:
    manifest_payload: JsonMap
    lammps_contract_payload: JsonMap
    incident_schedule_payload: JsonMap
    surface_state_payload: JsonMap


def campaign_job_manifest_payload(
    campaign: JsonMap,
    request: JsonMap,
    descriptor_root: Path,
) -> JsonMap:
    return stage_md_campaign(campaign, request, descriptor_root).manifest_payload


def stage_md_campaign(
    campaign: JsonMap,
    request: JsonMap,
    descriptor_root: Path,
    incident_count: int = 500,
) -> MDCampaignStagingBundle:
    try:
        material = build_material_state(
            material_id=as_str(require(campaign, "material_id"), "material_id"),
            phases=_phases(campaign),
            descriptor_root=descriptor_root,
            method="fixture",
            pr_selectivity=_pr_selectivity(request),
        )
        contract = build_lammps_output_contract(_contract_run_id(request), material.force_field)
        schedule_payload = incident_schedule_payload(campaign, request, incident_count)
        state_payload = surface_state_payload(campaign, request)
    except (MaterialBuilderError, LAMMPSContractError, MDCampaignInputError) as exc:
        raise MDCampaignStagingError(str(exc)) from exc

    contract_payload = contract.manifest_payload()
    manifest_payload = _campaign_payload(campaign, request) | _preflight_payload(contract_payload)
    physics_report = assess_md_physics_readiness(
        manifest_payload,
        contract_payload,
        schedule_payload,
        state_payload,
    )
    return MDCampaignStagingBundle(
        manifest_payload=manifest_payload | _physics_gate_payload(physics_report),
        lammps_contract_payload=contract_payload,
        incident_schedule_payload=schedule_payload,
        surface_state_payload=state_payload,
    )


def _preflight_payload(contract_payload: JsonMap) -> JsonMap:
    return {
        "preflight_gate": "lammps_contract_ready",
        "unit_style": contract_payload["unit_style"],
        "energy_unit": contract_payload["energy_unit"],
        "required_outputs": contract_payload["required_outputs"],
        "zbl_required": contract_payload["zbl_required"],
        "high_energy_collision_model": contract_payload["high_energy_collision_model"],
        "force_field_protocol_id": contract_payload["force_field_protocol_id"],
        "force_field_source_url": contract_payload["force_field_source_url"],
    }


def _physics_gate_payload(report: MDPhysicsReadinessReport) -> JsonMap:
    return {
        "physics_gate_status": report.payload["gate_status"],
        "physics_production_ready": report.production_ready,
        "physics_evidence": report.payload["evidence"],
        "physics_blockers": report.payload["blockers"],
        "physics_errors": report.payload["errors"],
    }


def _campaign_payload(campaign: JsonMap, request: JsonMap) -> JsonMap:
    energy = as_mapping(require(campaign, "energy_strata"), "energy_strata")
    polar = as_mapping(require(campaign, "polar_strata"), "polar_strata")
    azimuth = as_mapping(require(campaign, "azimuth_strata"), "azimuth_strata")
    return {
        "run_status": "dry_run_ready",
        "request_id": as_str(require(request, "request_id"), "request_id"),
        "protocol_id": as_str(require(campaign, "protocol_id"), "protocol_id"),
        "material_id": as_str(require(campaign, "material_id"), "material_id"),
        "ion_species": as_str(require(campaign, "ion_species"), "ion_species"),
        "phases": list(_phases(campaign)),
        "energy_range_eV": _range_payload(energy),
        "polar_range_deg": _range_payload(polar),
        "azimuth_range_deg": _range_payload(azimuth),
    }


def _contract_run_id(request: JsonMap) -> str:
    return f"{as_str(require(request, 'request_id'), 'request_id')}-lammps-contract"


def _phases(campaign: JsonMap) -> tuple[str, ...]:
    return tuple(
        as_str(item, "phases")
        for item in as_sequence(require(campaign, "phases"), "phases")
    )


def _range_payload(strata: JsonMap) -> list[float]:
    return [
        as_float(require(strata, "minimum"), "minimum"),
        as_float(require(strata, "maximum"), "maximum"),
    ]


def _pr_selectivity(request: JsonMap) -> float:
    scene = as_mapping(require(request, "scene"), "scene")
    stack = as_mapping(require(scene, "material_stack"), "material_stack")
    materials = as_sequence(require(stack, "materials"), "materials")
    for material_value in materials:
        material = as_mapping(material_value, "materials[]")
        if _is_pr_mask(material):
            return as_float(require(material, "pr_selectivity"), "pr_selectivity")
    raise MDCampaignStagingError("pr_selectivity_required")


def _is_pr_mask(material: JsonMap) -> bool:
    return (
        as_str(require(material, "role"), "role") == "mask"
        and as_str(require(material, "material_id"), "material_id") == "PR"
    )
