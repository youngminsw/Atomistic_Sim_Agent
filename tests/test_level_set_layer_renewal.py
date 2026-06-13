from __future__ import annotations

import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _transport_field():
    from sim_agent.transport import TransportCell, TransportCellKey, TransportField

    return TransportField(
        mode="2d",
        feature_type="trench",
        cells=(
            TransportCell(
                key=TransportCellKey(ix=3, iy=4, iz=0),
                material_id="Si",
                region="opening",
                hit_count=2,
                deposited_energy_eV=130.0,
                removed_depth_nm=0.04,
                damage_dose=2.5,
                roughness_rms_nm=0.16,
                implanted_inert_fraction=0.02,
                local_fluence=2.0,
                event_ids=("transport-ion-000000", "transport-ion-000001"),
            ),
        ),
    )


def _surface_state():
    from sim_agent.schemas.state import SurfaceState

    return SurfaceState(
        material_id="Si",
        phase="crystal",
        amorphous_index=0.1,
        damage_dose=2.0,
        roughness_rms_nm=0.16,
        roughness_corr_length_nm=1.0,
        implanted_inert_fraction=0.02,
        local_fluence=2.0,
        removed_depth_nm=0.04,
        rdf_crystal_similarity=0.8,
        rdf_amorphous_similarity=0.2,
        coordination_defect_fraction=0.05,
        active_layer_thickness_nm=0.03,
        kernel_version="fixture-kernel",
    )


def _fresh_surface_state():
    surface = _surface_state()
    return type(surface)(
        material_id=surface.material_id,
        phase=surface.phase,
        amorphous_index=surface.amorphous_index,
        damage_dose=0.0,
        roughness_rms_nm=0.1,
        roughness_corr_length_nm=surface.roughness_corr_length_nm,
        implanted_inert_fraction=0.0,
        local_fluence=0.0,
        removed_depth_nm=0.0,
        rdf_crystal_similarity=surface.rdf_crystal_similarity,
        rdf_amorphous_similarity=surface.rdf_amorphous_similarity,
        coordination_defect_fraction=0.0,
        active_layer_thickness_nm=surface.active_layer_thickness_nm,
        kernel_version=surface.kernel_version,
    )


def _volume_state():
    from sim_agent.schemas.state import VolumeState

    return VolumeState(
        material_id="Si",
        phase="amorphous",
        initial_amorphous_index=0.85,
        density_factor=0.96,
        preexisting_damage=0.2,
        implanted_inert_fraction=0.01,
        rdf_order_features={"crystal_similarity": 0.2, "amorphous_similarity": 0.8},
        grain_or_orientation_id=None,
        source_structure_id="amorphous-underlayer-001",
    )


def test_transport_field_evolves_2d_trench_profile_over_time() -> None:
    from sim_agent.level_set import LevelSetConfig, evolve_transport_profile

    timeline = evolve_transport_profile(_transport_field(), LevelSetConfig(time_steps=4, time_step_s=0.1, cell_area_nm2=2.0))
    final_cell = timeline.final_state.cells[0]

    assert timeline.mode == "2d"
    assert timeline.feature_type.value == "trench"
    assert timeline.state_count == 5
    assert final_cell.surface_depth_nm == pytest.approx(0.04)
    assert final_cell.cumulative_energy_eV == pytest.approx(130.0)
    assert final_cell.event_ids == ("transport-ion-000000", "transport-ion-000001")


def test_layer_renewal_initializes_surface_from_next_volume_state() -> None:
    from sim_agent.level_set import renew_surface_state_if_needed

    result = renew_surface_state_if_needed(_surface_state(), (_volume_state(),))

    assert result.renewed is True
    assert result.surface_state.phase == "amorphous"
    assert result.surface_state.amorphous_index == pytest.approx(0.85)
    assert result.surface_state.damage_dose == pytest.approx(0.2)
    assert result.surface_state.local_fluence == 0.0
    assert result.surface_state.removed_depth_nm == 0.0
    assert result.surface_state.rdf_amorphous_similarity == pytest.approx(0.8)


def test_level_set_marks_cells_renewed_when_active_layer_is_consumed() -> None:
    from sim_agent.level_set import LevelSetConfig, evolve_transport_profile_with_layer_renewal

    timeline = evolve_transport_profile_with_layer_renewal(
        _transport_field(),
        LevelSetConfig(time_steps=4, time_step_s=0.1, cell_area_nm2=2.0),
        _fresh_surface_state(),
        (_volume_state(),),
    )
    final_cell = timeline.final_state.cells[0]

    assert timeline.renewal_count == 1
    assert final_cell.layer_renewed is True
    assert final_cell.surface_phase == "amorphous"
