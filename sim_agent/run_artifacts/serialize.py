from __future__ import annotations

from sim_agent.geometry import PatternGeometry3D
from sim_agent.level_set import ProfileDiagnostic, ProfileState, ProfileTimeline
from sim_agent.schemas._parse import JsonMap


def timeline_payload(timeline: ProfileTimeline) -> JsonMap:
    return {
        "feature_type": timeline.feature_type.value,
        "state_count": timeline.state_count,
        "cell_area_nm2": _round(timeline.cell_area_nm2),
        "process": timeline.process or {},
        "final_removed_volume_nm3": _round(timeline.final_state.total_removed_volume_nm3),
        "states": [_state_payload(state) for state in timeline.states],
    }


def diagnostics_payload(
    geometry: PatternGeometry3D,
    timeline: ProfileTimeline,
    click_points_nm: tuple[tuple[float, float, float], ...],
) -> JsonMap:
    diagnostics = tuple(timeline.diagnostic_at_nm(geometry, x_nm, y_nm, z_nm) for x_nm, y_nm, z_nm in click_points_nm)
    return {
        "click_count": len(diagnostics),
        "clicks": [_diagnostic_payload(item, click_points_nm[index]) for index, item in enumerate(diagnostics)],
    }


def manifest_payload(
    run_id: str,
    geometry: PatternGeometry3D,
    timeline: ProfileTimeline,
) -> JsonMap:
    manifest = geometry.export_manifest()
    return {
        "run_id": run_id,
        "feature_type": manifest.feature_type.value,
        "target_material_id": manifest.target_material_id,
        "mask_material_id": manifest.mask_material_id,
        "pr_selectivity": _round(manifest.pr_selectivity),
        "state_count": timeline.state_count,
        "final_removed_volume_nm3": _round(timeline.final_state.total_removed_volume_nm3),
        "artifact_types": ["run_manifest", "profile_timeline", "click_diagnostics"],
        "artifacts": {
            "manifest": "manifest.json",
            "timeline": "timeline.json",
            "diagnostics": "diagnostics.json",
        },
    }


def _state_payload(state: ProfileState) -> JsonMap:
    return {
        "step_index": state.step_index,
        "time_s": _round(state.time_s),
        "total_removed_volume_nm3": _round(state.total_removed_volume_nm3),
        "cells": [
            {
                "ix": cell.key.ix,
                "iy": cell.key.iy,
                "iz": cell.key.iz,
                "material_id": cell.material_id,
                "region": cell.region,
                "surface_depth_nm": _round(cell.surface_depth_nm),
                "cumulative_energy_eV": _round(cell.cumulative_energy_eV),
                "removal_law": cell.removal_law,
                "event_ids": list(cell.event_ids),
            }
            for cell in state.cells
        ],
    }


def _diagnostic_payload(diagnostic: ProfileDiagnostic, click_nm: tuple[float, float, float]) -> JsonMap:
    return {
        "click_nm": [_round(value) for value in click_nm],
        "ix": diagnostic.key.ix,
        "iy": diagnostic.key.iy,
        "iz": diagnostic.key.iz,
        "material_id": diagnostic.material_id,
        "region": diagnostic.region,
        "depth_history_nm": [_round(value) for value in diagnostic.depth_history_nm],
        "energy_history_eV": [_round(value) for value in diagnostic.energy_history_eV],
        "removal_law": diagnostic.removal_law,
        "event_ids": list(diagnostic.event_ids),
    }


def _round(value: float) -> float:
    return round(value, 6)
