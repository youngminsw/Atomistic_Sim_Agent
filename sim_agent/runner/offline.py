from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from sim_agent.geometry import (
    GridShape,
    PatternGeometry2D,
    PatternGeometry3D,
    load_pattern_geometry_2d,
    load_pattern_geometry_from_scene,
)
from sim_agent.kmc import CellKey
from sim_agent.level_set import LevelSetConfig, ProfileTimeline, evolve_transport_profile
from sim_agent.ml_surrogate import (
    InteractionKernel,
    InteractionKernelRegistry,
    KernelFeatureSpec,
    build_fixture_interaction_kernel,
)
from sim_agent.run_artifacts import timeline_payload
from sim_agent.schemas._parse import JsonMap, as_mapping
from sim_agent.schemas.distributions import IonAngularDistribution, IonEnergyBin, IonEnergyDistribution
from sim_agent.transport import (
    FeatureScaleProcessSchedule,
    TransportCell,
    TransportField,
    TransportHitRecord,
    TransportResult,
    process_schedule_payload,
    run_transport_2d,
    run_transport_3d,
)

from .artifacts import (
    complete_manifest_payload,
    failed_manifest_payload,
    hit_history_payload,
    qa_report_payload,
    transport_field_payload,
    write_json,
)
from .active_learning import write_active_learning_artifacts
from .surrogate_artifacts import write_surrogate_artifact_bundle
from .types import OfflineRunRequest, OfflineRunResult, RunManagerError


def run_offline_simulation(request: OfflineRunRequest) -> OfflineRunResult:
    paths = _paths(request.output_dir)
    if not request.kernel_path.exists():
        write_json(paths.manifest, failed_manifest_payload(request.run_id, request.mode, "kernel_not_found"))
        return _result(request, "failed", 1, "kernel_not_found")
    kernel = _kernel(request.kernel_path, request.events_path)
    surrogate = write_surrogate_artifact_bundle(request.output_dir, kernel)
    geometry = _geometry(request)
    schedule = FeatureScaleProcessSchedule(
        duration_s=request.process_duration_s,
        flux_ions_cm2_s=request.flux_ions_cm2_s,
        active_area_nm2=_active_area_nm2(geometry),
        sampled_ion_count=request.ion_count,
    )
    process = process_schedule_payload(schedule)
    transport = _run_transport(request, kernel.registry(), geometry)
    timeline = evolve_transport_profile(
        transport.field,
        LevelSetConfig(
            time_steps=request.time_steps,
            time_step_s=schedule.duration_s / request.time_steps,
            cell_area_nm2=request.cell_area_nm2,
        ),
        process=process,
    )
    click_index = _click_index_payload(timeline, transport.field, transport.hit_history)
    write_json(
        paths.manifest,
        complete_manifest_payload(
            request.run_id,
            request.mode,
            transport.feature_type,
            timeline,
            request.ion_count,
            process,
            surrogate.payload,
        ),
    )
    write_json(paths.timeline, timeline_payload(timeline))
    write_json(paths.transport_field, transport_field_payload(transport.field))
    write_json(paths.hit_history, hit_history_payload(transport.hit_history))
    write_json(paths.click_index, click_index)
    write_active_learning_artifacts(request.output_dir, request.run_id, kernel.manifest, transport.hit_history)
    write_json(
        paths.qa_report,
        qa_report_payload(
            request.run_id,
            request.mode,
            transport.feature_type,
            timeline,
            transport.field,
            transport.hit_history,
            process,
            surrogate.payload,
        ),
    )
    return _result(request, "complete", 11, "")


def _run_transport(
    request: OfflineRunRequest,
    registry: InteractionKernelRegistry,
    geometry: PatternGeometry2D | PatternGeometry3D,
) -> TransportResult:
    match request.mode:
        case "3d":
            if not isinstance(geometry, PatternGeometry3D):
                raise RunManagerError("geometry_3d_required")
            return run_transport_3d(
                geometry=geometry,
                registry=registry,
                energy_distribution=_energy_distribution(request),
                angular_distribution=_angular_distribution(request),
                ion_count=request.ion_count,
                seed=request.seed,
                duration_s=request.process_duration_s,
            )
        case "2d":
            if not isinstance(geometry, PatternGeometry2D):
                raise RunManagerError("geometry_2d_required")
            return run_transport_2d(
                geometry=geometry,
                registry=registry,
                energy_distribution=_energy_distribution(request),
                angular_distribution=_angular_distribution(request),
                ion_count=request.ion_count,
                seed=request.seed,
                duration_s=request.process_duration_s,
            )
        case unreachable:
            assert_never(unreachable)


def _kernel(kernel_path: Path, events_path: Path) -> InteractionKernel:
    spec = KernelFeatureSpec.from_mapping(as_mapping(json.loads(kernel_path.read_text(encoding="utf-8")), "kernel"))
    return build_fixture_interaction_kernel(events_path, spec, provenance_source=str(events_path))


def _geometry(request: OfflineRunRequest) -> PatternGeometry2D | PatternGeometry3D:
    match request.mode:
        case "3d":
            return _geometry_3d(request)
        case "2d":
            return _geometry_2d(request)
        case unreachable:
            assert_never(unreachable)


def _geometry_3d(request: OfflineRunRequest) -> PatternGeometry3D:
    if request.scene_path is None:
        raise RunManagerError("scene_path_required")
    scene = as_mapping(json.loads(request.scene_path.read_text(encoding="utf-8")), "scene")
    return load_pattern_geometry_from_scene(scene, request.source_root, GridShape(32, 32, 16), target_depth_nm=24.0)


def _geometry_2d(request: OfflineRunRequest) -> PatternGeometry2D:
    if request.image_path is None:
        raise RunManagerError("image_path_required")
    return load_pattern_geometry_2d(
        request.image_path,
        pixel_size_nm=request.pixel_size_nm,
        target_material_id="Si",
        mask_material_id="PR",
        structure_description="offline 2D PR trench image",
    )


def _active_area_nm2(geometry: PatternGeometry2D | PatternGeometry3D) -> float:
    if isinstance(geometry, PatternGeometry3D):
        return geometry.bounds.x_span_nm * geometry.bounds.y_span_nm
    return geometry.width_nm * 1.0


def _energy_distribution(request: OfflineRunRequest) -> IonEnergyDistribution:
    if request.energy_distribution is not None:
        return request.energy_distribution
    return IonEnergyDistribution(
        kind="histogram",
        unit="eV",
        bins=(
            IonEnergyBin(min=80.0, max=90.0, probability=0.5),
            IonEnergyBin(min=90.0, max=100.0, probability=0.5),
        ),
    )


def _angular_distribution(request: OfflineRunRequest) -> IonAngularDistribution:
    if request.angular_distribution is not None:
        return request.angular_distribution
    return IonAngularDistribution(
        kind="uniform",
        polar_min_deg=30.0,
        polar_max_deg=45.0,
        azimuth_min_deg=120.0,
        azimuth_max_deg=240.0,
    )


def _click_index_payload(
    timeline: ProfileTimeline,
    field: TransportField,
    hit_history: tuple[TransportHitRecord, ...],
) -> JsonMap:
    clicks = tuple(_click_payload(timeline, cell, hit_history) for cell in field.cells)
    return {
        "click_count": len(clicks),
        "process": timeline.process or {},
        "clicks": list(clicks),
    }


def _click_payload(
    timeline: ProfileTimeline,
    field_cell: TransportCell,
    hit_history: tuple[TransportHitRecord, ...],
) -> JsonMap:
    key = CellKey(field_cell.key.ix, field_cell.key.iy, field_cell.key.iz)
    cells = tuple(state.cell_at_key(key) for state in timeline.states)
    matching_hits = tuple(hit for hit in hit_history if hit.event_id in field_cell.event_ids)
    first_hit = matching_hits[0]
    return {
        "ix": key.ix,
        "iy": key.iy,
        "iz": key.iz,
        "click_nm": [round(first_hit.x_nm, 6), round(first_hit.y_nm, 6), round(first_hit.z_nm, 6)],
        "material_id": field_cell.material_id,
        "region": field_cell.region,
        "energy_transfer_eV": round(field_cell.deposited_energy_eV, 6),
        "damage_dose": round(field_cell.damage_dose, 6),
        "removed_depth_nm": round(field_cell.removed_depth_nm, 6),
        "profile_history_nm": [round(cell.surface_depth_nm, 6) for cell in cells],
        "energy_history_eV": [round(cell.cumulative_energy_eV, 6) for cell in cells],
        "incident_history": [_incident_payload(hit) for hit in matching_hits],
        "uncertainty_ood": any(hit.uncertainty_ood for hit in matching_hits),
        "event_ids": list(field_cell.event_ids),
    }


def _incident_payload(hit: TransportHitRecord) -> JsonMap:
    return {
        "event_id": hit.event_id,
        "time_step": hit.time_step,
        "time_s": round(hit.time_s, 6),
        "energy_eV": round(hit.energy_eV, 6),
        "polar_deg": round(hit.polar_deg, 6),
        "azimuth_deg": round(hit.azimuth_deg, 6),
        "local_incidence_deg": round(hit.local_incidence_deg, 6),
        "deposited_energy_eV": round(hit.deposited_energy_eV, 6),
        "removed_depth_nm": round(hit.removed_depth_nm, 6),
        "uncertainty_ood": hit.uncertainty_ood,
        "uncertainty_score": round(hit.uncertainty_score, 6),
    }


def _result(request: OfflineRunRequest, status: str, artifact_count: int, reason: str) -> OfflineRunResult:
    paths = _paths(request.output_dir)
    return OfflineRunResult(
        run_id=request.run_id,
        run_status=status,
        output_dir=request.output_dir,
        manifest_path=paths.manifest,
        timeline_path=paths.timeline,
        transport_field_path=paths.transport_field,
        hit_history_path=paths.hit_history,
        click_index_path=paths.click_index,
        uncertainty_map_path=paths.uncertainty_map,
        active_learning_plan_path=paths.active_learning_plan,
        qa_report_path=paths.qa_report,
        artifact_count=artifact_count,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _Paths:
    manifest: Path
    timeline: Path
    transport_field: Path
    hit_history: Path
    click_index: Path
    uncertainty_map: Path
    active_learning_plan: Path
    qa_report: Path


def _paths(output_dir: Path) -> _Paths:
    return _Paths(
        manifest=output_dir / "manifest.json",
        timeline=output_dir / "timeline.json",
        transport_field=output_dir / "transport_field.json",
        hit_history=output_dir / "hit_history.json",
        click_index=output_dir / "click_index.json",
        uncertainty_map=output_dir / "uncertainty_map.json",
        active_learning_plan=output_dir / "active_learning_plan.json",
        qa_report=output_dir / "qa_report.json",
    )
