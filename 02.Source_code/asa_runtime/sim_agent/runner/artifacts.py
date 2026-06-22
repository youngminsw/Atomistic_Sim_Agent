from __future__ import annotations

import json
from pathlib import Path

from sim_agent.level_set import ProfileTimeline
from sim_agent.schemas._parse import JsonMap
from sim_agent.transport import TransportField, TransportHitRecord

from .artifact_contract import run_artifact_filename_map, run_artifact_types


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def transport_field_payload(field: TransportField) -> JsonMap:
    return {
        "mode": field.mode,
        "feature_type": field.feature_type,
        "cell_count": field.cell_count,
        "total_deposited_energy_eV": _round(field.total_deposited_energy_eV),
        "total_removed_depth_nm": _round(field.total_removed_depth_nm),
        "cells": [
            {
                "ix": cell.key.ix,
                "iy": cell.key.iy,
                "iz": cell.key.iz,
                "material_id": cell.material_id,
                "region": cell.region,
                "hit_count": cell.hit_count,
                "deposited_energy_eV": _round(cell.deposited_energy_eV),
                "damage_dose": _round(cell.damage_dose),
                "removed_depth_nm": _round(cell.removed_depth_nm),
                "local_fluence": _round(cell.local_fluence),
                "event_ids": list(cell.event_ids),
            }
            for cell in field.cells
        ],
    }


def hit_history_payload(hit_history: tuple[TransportHitRecord, ...]) -> JsonMap:
    return {
        "hit_history_count": len(hit_history),
        "hits": [
            {
                "event_id": hit.event_id,
                "time_step": hit.time_step,
                "time_s": _round(hit.time_s),
                "x_nm": _round(hit.x_nm),
                "y_nm": _round(hit.y_nm),
                "z_nm": _round(hit.z_nm),
                "material_id": hit.material_id,
                "region": hit.region,
                "energy_eV": _round(hit.energy_eV),
                "local_incidence_deg": _round(hit.local_incidence_deg),
                "deposited_energy_eV": _round(hit.deposited_energy_eV),
                "removed_depth_nm": _round(hit.removed_depth_nm),
                "uncertainty_ood": hit.uncertainty_ood,
                "uncertainty_score": _round(hit.uncertainty_score),
                "uncertainty_reason": hit.uncertainty_reason,
            }
            for hit in hit_history
        ],
    }


def complete_manifest_payload(
    run_id: str,
    mode: str,
    feature_type: str,
    timeline: ProfileTimeline,
    ion_count: int,
    process: JsonMap,
    surrogate: JsonMap,
) -> JsonMap:
    return {
        "run_id": run_id,
        "run_status": "complete",
        "mode": mode,
        "feature_type": feature_type,
        "ion_count": ion_count,
        "process": process,
        "surrogate": surrogate,
        "state_count": timeline.state_count,
        "final_removed_volume_nm3": _round(timeline.final_state.total_removed_volume_nm3),
        "artifact_types": run_artifact_types(),
        "artifacts": run_artifact_filename_map(),
        "verification_statuses": {
            "transport": "complete",
            "level_set": "complete",
            "click_index": "complete",
            "active_learning": "complete",
            "surrogate_training": "complete",
            "qa_report": "complete",
        },
    }


def qa_report_payload(
    run_id: str,
    mode: str,
    feature_type: str,
    timeline: ProfileTimeline,
    field: TransportField,
    hit_history: tuple[TransportHitRecord, ...],
    process: JsonMap,
    surrogate: JsonMap,
) -> JsonMap:
    blockers = _qa_blockers(timeline, field, hit_history, process, surrogate)
    return {
        "report_version": "demo_qa_agent_report_v1",
        "evidence_scope": "offline_demo_fixture",
        "agent_id": "qa_agent",
        "run_id": run_id,
        "mode": mode,
        "feature_type": feature_type,
        "status": "pass" if not blockers else "blocked",
        "hard_blockers": blockers,
        "checks": [
            {
                "check_id": "level_set_profile_timeline",
                "status": "pass" if timeline.state_count > 1 else "blocked",
                "evidence": {
                    "state_count": timeline.state_count,
                    "final_time_s": _round(timeline.final_state.time_s),
                    "final_removed_volume_nm3": _round(timeline.final_state.total_removed_volume_nm3),
                },
            },
            {
                "check_id": "position_resolved_energy",
                "status": "pass" if field.cell_count > 0 and field.total_deposited_energy_eV > 0.0 else "blocked",
                "evidence": {
                    "cell_count": field.cell_count,
                    "total_deposited_energy_eV": _round(field.total_deposited_energy_eV),
                },
            },
            {
                "check_id": "click_diagnostics",
                "status": "pass" if field.cell_count > 0 else "blocked",
                "evidence": {
                    "diagnostic_cell_count": field.cell_count,
                    "click_fields": [
                        "material_id",
                        "region",
                        "energy_transfer_eV",
                        "profile_history_nm",
                        "incident_history",
                    ],
                },
            },
            {
                "check_id": "process_time_scale",
                "status": "pass" if float(process.get("duration_s", 0.0)) > 0.0 else "blocked",
                "evidence": {
                    "duration_s": process.get("duration_s", 0.0),
                    "fluence_ions_cm2": process.get("fluence_ions_cm2", 0.0),
                    "physical_incident_count": process.get("physical_incident_count", 0.0),
                },
            },
            {
                "check_id": "incident_history",
                "status": "pass" if hit_history else "blocked",
                "evidence": {
                    "hit_count": len(hit_history),
                    "first_event_id": hit_history[0].event_id if hit_history else "",
                },
            },
            {
                "check_id": "surrogate_training_gate",
                "status": "pass" if _surrogate_ok(surrogate) else "blocked",
                "evidence": {
                    "training_backend": surrogate.get("training_backend", ""),
                    "quality_gate_decision": surrogate.get("quality_gate_decision", ""),
                    "training_event_count": surrogate.get("training_event_count", 0),
                    "registered_for_feature_scale": surrogate.get("registered_for_feature_scale", False),
                },
            },
        ],
        "final_recommendation": "accept_demo_artifacts" if not blockers else "block_until_fixed",
    }


def failed_manifest_payload(run_id: str, mode: str, reason: str) -> JsonMap:
    return {
        "run_id": run_id,
        "run_status": "failed",
        "mode": mode,
        "reason": reason,
        "artifact_types": ["run_manifest"],
        "artifacts": {"manifest": "manifest.json"},
        "verification_statuses": {"run_manager": "failed"},
    }


def _round(value: float) -> float:
    return round(value, 6)


def _qa_blockers(
    timeline: ProfileTimeline,
    field: TransportField,
    hit_history: tuple[TransportHitRecord, ...],
    process: JsonMap,
    surrogate: JsonMap,
) -> list[str]:
    blockers: list[str] = []
    if timeline.state_count <= 1:
        blockers.append("level_set_profile_timeline_missing")
    if field.cell_count <= 0 or field.total_deposited_energy_eV <= 0.0:
        blockers.append("position_resolved_energy_missing")
    if not hit_history:
        blockers.append("incident_history_missing")
    if float(process.get("duration_s", 0.0)) <= 0.0:
        blockers.append("process_time_scale_missing")
    if not _surrogate_ok(surrogate):
        blockers.append("surrogate_training_gate_not_accepted")
    return blockers


def _surrogate_ok(surrogate: JsonMap) -> bool:
    return bool(surrogate.get("quality_gate_accepted")) and bool(surrogate.get("registered_for_feature_scale"))
