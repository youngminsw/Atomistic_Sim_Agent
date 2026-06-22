from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class AgentCliRequestInput:
    goal: str
    material: str
    phase: str
    ion: str
    feature_type: str
    mode: str
    flux_ions_cm2_s: float
    active_layer_thickness_nm: float
    pr_selectivity: float
    md_box_x_nm: float
    md_box_y_nm: float
    md_box_mobile_depth_nm: float
    md_box_fixed_depth_nm: float
    md_box_thermostat_depth_nm: float
    md_box_expected_cascade_depth_nm: float
    md_box_atom_count: int
    md_timestep_fs: float
    md_run_length_ps: float
    lammps_structure_source: str | None
    lammps_structure_preparation: str
    model_provider: str = "openai-codex"
    model_name: str = "gpt-5-codex"
    reasoning_effort: str = "high"
    model_base_url: str = "https://model-gateway.local/v1"
    model_auth_mode: str = "gateway"
    model_api_key_env: str = "MODEL_GATEWAY_TOKEN"


def parse_range(raw: str, field: str) -> tuple[float, float]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise SchemaValidationError(f"{field} must use min:max")
    low = _parse_float(parts[0], f"{field}.minimum")
    high = _parse_float(parts[1], f"{field}.maximum")
    if high <= low:
        raise SchemaValidationError(f"{field} maximum must be greater than minimum")
    return (low, high)


def build_agent_cli_request(
    config: AgentCliRequestInput,
    energy_range: tuple[float, float],
    polar_range: tuple[float, float],
    azimuth_range: tuple[float, float],
) -> JsonMap:
    return {
        "request_id": _request_id(config),
        "user_goal": config.goal,
        "llm_endpoint": _model_provider(config),
        "scene": _scene(config),
        "recipe": _recipe(config, energy_range, polar_range, azimuth_range),
    }


def _parse_float(raw: str, field: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise SchemaValidationError(f"{field} must be numeric") from exc


def _request_id(config: AgentCliRequestInput) -> str:
    return (
        f"cli_{config.ion.lower()}_{config.material.lower()}_"
        f"{config.phase}_{config.feature_type}"
    )


def _model_provider(config: AgentCliRequestInput) -> JsonMap:
    return {
        "provider": config.model_provider,
        "model": config.model_name,
        "reasoning_effort": config.reasoning_effort,
        "base_url": config.model_base_url,
        "auth_mode": config.model_auth_mode,
        "api_key_env": config.model_api_key_env,
    }


def _scene(config: AgentCliRequestInput) -> JsonMap:
    return {
        "scene_id": f"{config.feature_type}_{config.material}_{config.phase}_scene",
        "mode": config.mode,
        "feature_type": config.feature_type,
        "geometry_source": _geometry_source(config.mode, config.feature_type),
        "material_stack": _material_stack(
            config.material,
            config.phase,
            config.pr_selectivity,
        ),
        "volume_states": [_volume_state(config.material, config.phase)],
        "surface_state": _surface_state(
            config.material,
            config.phase,
            config.active_layer_thickness_nm,
            config,
        ),
    }


def _geometry_source(mode: str, feature_type: str) -> JsonMap:
    match (mode, feature_type):
        case ("3d", "hole"):
            path = "fixtures/geometry/pr_hole.stl"
            kind = "mesh"
        case ("3d", "trench"):
            path = "fixtures/geometry/pr_trench.stl"
            kind = "mesh"
        case ("2d", "hole"):
            path = "fixtures/geometry/pr_hole_mask.png"
            kind = "image"
        case ("2d", "trench"):
            path = "fixtures/geometry/pr_trench.png"
            kind = "image"
        case unreachable:
            assert_never(unreachable)
    return {"kind": kind, "path": path, "units": "nm"}


def _material_stack(material: str, phase: str, pr_selectivity: float) -> JsonMap:
    return {
        "materials": [
            {
                "material_id": "PR",
                "role": "mask",
                "phase": "amorphous",
                "composition": {"C": 8.0, "H": 8.0},
                "pr_selectivity": pr_selectivity,
            },
            {
                "material_id": material,
                "role": "target",
                "phase": phase,
                "composition": {material: 1.0},
            },
        ]
    }


def _volume_state(material: str, phase: str) -> JsonMap:
    return {
        "material_id": material,
        "phase": phase,
        "initial_amorphous_index": _amorphous_index(phase),
        "density_factor": 0.98 if phase == "amorphous" else 1.0,
        "preexisting_damage": 0.0,
        "implanted_inert_fraction": 0.0,
        "rdf_order_features": _rdf_features(phase),
        "grain_or_orientation_id": None if phase == "amorphous" else f"{material}_100",
        "source_structure_id": _source_structure_id(material, phase),
    }


def _surface_state(
    material: str,
    phase: str,
    active_layer_thickness_nm: float,
    config: AgentCliRequestInput,
) -> JsonMap:
    payload: JsonMap = {
        "material_id": material,
        "phase": phase,
        "amorphous_index": _amorphous_index(phase),
        "damage_dose": 0.0,
        "roughness_rms_nm": 0.1,
        "roughness_corr_length_nm": 2.0,
        "implanted_inert_fraction": 0.0,
        "local_fluence": 0.0,
        "removed_depth_nm": 0.0,
        "rdf_crystal_similarity": 0.12 if phase == "amorphous" else 0.98,
        "rdf_amorphous_similarity": 0.88 if phase == "amorphous" else 0.02,
        "coordination_defect_fraction": 0.0,
        "active_layer_thickness_nm": active_layer_thickness_nm,
        "kernel_version": f"Ar_on_{material}__physical_v001",
        "md_box": {
            "x_nm": config.md_box_x_nm,
            "y_nm": config.md_box_y_nm,
            "mobile_depth_nm": config.md_box_mobile_depth_nm,
            "fixed_depth_nm": config.md_box_fixed_depth_nm,
            "thermostat_depth_nm": config.md_box_thermostat_depth_nm,
            "expected_cascade_depth_nm": config.md_box_expected_cascade_depth_nm,
            "atom_count": config.md_box_atom_count,
            "timestep_fs": config.md_timestep_fs,
            "run_length_ps": config.md_run_length_ps,
        },
    }
    if config.lammps_structure_source:
        payload["lammps_structure_source"] = {
            "kind": "user_supplied",
            "path": config.lammps_structure_source,
            "phase": phase,
            "preparation": config.lammps_structure_preparation,
        }
    return payload


def _rdf_features(phase: str) -> JsonMap:
    if phase == "amorphous":
        return {
            "first_peak_nm": 0.237,
            "crystal_similarity": 0.12,
            "amorphous_similarity": 0.88,
        }
    return {
        "first_peak_nm": 0.235,
        "crystal_similarity": 0.98,
        "amorphous_similarity": 0.02,
    }


def _source_structure_id(material: str, phase: str) -> str:
    if phase == "amorphous":
        return f"a_{material.lower()}_melt_quench_relaxed"
    return f"{material.lower()}_100_relaxed"


def _amorphous_index(phase: str) -> float:
    return 1.0 if phase == "amorphous" else 0.0


def _recipe(
    config: AgentCliRequestInput,
    energy_range: tuple[float, float],
    polar_range: tuple[float, float],
    azimuth_range: tuple[float, float],
) -> JsonMap:
    return {
        "ion_species": config.ion,
        "ion_energy_distribution": _energy_distribution(energy_range),
        "ion_angular_distribution": _angular_distribution(polar_range, azimuth_range),
        "flux_schedule": _flux_schedule(config.flux_ions_cm2_s),
        "species_mix": [{"species": config.ion, "fraction": 1.0}],
    }


def _energy_distribution(energy_range: tuple[float, float]) -> JsonMap:
    return {
        "kind": "histogram",
        "unit": "eV",
        "bins": [{"min": energy_range[0], "max": energy_range[1], "probability": 1.0}],
    }


def _angular_distribution(
    polar_range: tuple[float, float],
    azimuth_range: tuple[float, float],
) -> JsonMap:
    return {
        "kind": "uniform",
        "polar_min_deg": polar_range[0],
        "polar_max_deg": polar_range[1],
        "azimuth_min_deg": azimuth_range[0],
        "azimuth_max_deg": azimuth_range[1],
    }


def _flux_schedule(flux_ions_cm2_s: float) -> JsonMap:
    return {
        "unit": "ions_cm2_s",
        "segments": [{"start_s": 0.0, "end_s": 1.0, "flux": flux_ions_cm2_s}],
    }
