from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas.state import SurfaceState, VolumeState

from .types import LevelSetError


@dataclass(frozen=True, slots=True)
class LayerRenewalResult:
    renewed: bool
    surface_state: SurfaceState


def renew_surface_state_if_needed(
    surface_state: SurfaceState,
    volume_states: tuple[VolumeState, ...],
    additional_removed_depth_nm: float = 0.0,
) -> LayerRenewalResult:
    consumed_depth_nm = surface_state.removed_depth_nm + additional_removed_depth_nm
    if consumed_depth_nm < surface_state.active_layer_thickness_nm:
        return LayerRenewalResult(renewed=False, surface_state=surface_state)
    if not volume_states:
        raise LevelSetError("next_volume_state_required")
    next_volume = volume_states[0]
    return LayerRenewalResult(
        renewed=True,
        surface_state=SurfaceState(
            material_id=next_volume.material_id,
            phase=next_volume.phase,
            amorphous_index=next_volume.initial_amorphous_index,
            damage_dose=next_volume.preexisting_damage,
            roughness_rms_nm=0.0,
            roughness_corr_length_nm=surface_state.roughness_corr_length_nm,
            implanted_inert_fraction=next_volume.implanted_inert_fraction,
            local_fluence=0.0,
            removed_depth_nm=0.0,
            rdf_crystal_similarity=next_volume.rdf_order_features.get("crystal_similarity", 0.0),
            rdf_amorphous_similarity=next_volume.rdf_order_features.get("amorphous_similarity", 0.0),
            coordination_defect_fraction=0.0,
            active_layer_thickness_nm=surface_state.active_layer_thickness_nm,
            kernel_version=surface_state.kernel_version,
        ),
    )
