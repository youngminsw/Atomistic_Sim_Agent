from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_mapping

from .box_gate import assess_md_box_readiness


PRODUCTION_MIN_INCIDENTS: Final = 500


@dataclass(frozen=True, slots=True)
class MDProductionReadinessReport:
    production_ready: bool
    payload: JsonMap


def assess_md_production_readiness(
    request_payload: JsonMap,
    compute_response_payload: JsonMap,
    minimum_incidents: int = PRODUCTION_MIN_INCIDENTS,
) -> MDProductionReadinessReport:
    surface = _surface_state(request_payload)
    phase = _text(surface, "phase")
    incident_count = _incident_count(compute_response_payload)
    md_box_report = assess_md_box_readiness(surface.get("md_box"))
    blockers = list(md_box_report.blockers)
    evidence = list(md_box_report.evidence)

    if incident_count < minimum_incidents:
        blockers.append(f"incident_count_too_low:{incident_count}<{minimum_incidents}")
    else:
        evidence.append("production_incident_count_sufficient")
    if phase == "amorphous" and "lammps_structure_source" not in surface:
        blockers.append("amorphous_lammps_structure_source_required")
    if phase == "amorphous" and "lammps_structure_source" in surface:
        evidence.append("amorphous_structure_source_present")
    if phase == "crystal":
        evidence.append("crystal_structure_fixture_available")

    production_ready = not blockers
    return MDProductionReadinessReport(
        production_ready=production_ready,
        payload={
            "gate_status": "production_ready" if production_ready else "blocked",
            "production_ready": production_ready,
            "phase": phase,
            "incident_count": incident_count,
            "minimum_incidents": minimum_incidents,
            "structure_source_required": phase == "amorphous",
            "structure_source_present": "lammps_structure_source" in surface,
            "md_box": md_box_report.payload,
            "qa_gates": {
                "worker_capability": "required_before_lammps_execute",
                "slurm_job_script": "qa_required_before_submit",
                "actual_500_incident_execution": "required_before_downstream",
            },
            "evidence": evidence,
            "hard_blockers": blockers,
        },
    )


def _surface_state(request_payload: JsonMap) -> JsonMap:
    scene = as_mapping(request_payload.get("scene"), "scene")
    return as_mapping(scene.get("surface_state"), "surface_state")


def _incident_count(compute_response_payload: JsonMap) -> int:
    job = compute_response_payload.get("job")
    if not isinstance(job, dict):
        return 0
    command = job.get("command")
    if not isinstance(command, list | tuple):
        return 0
    return _incident_count_from_command(command)


def _incident_count_from_command(command: Sequence[object]) -> int:
    for index, value in enumerate(command):
        if value == "--incident-count" and index + 1 < len(command):
            return _positive_int(command[index + 1])
    return 0


def _positive_int(value: object) -> int:
    try:
        parsed = int(str(value))
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""
