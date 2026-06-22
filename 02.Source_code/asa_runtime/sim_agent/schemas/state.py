from __future__ import annotations

from dataclasses import dataclass

from ._parse import JsonMap, as_mapping, as_sequence, float_field, float_map, optional_str, str_field


@dataclass(frozen=True, slots=True)
class MaterialModel:
    material_id: str
    role: str
    phase: str
    composition: dict[str, float]
    pr_selectivity: float | None = None

    @classmethod
    def from_mapping(cls, value: JsonMap) -> MaterialModel:
        pr_value = value.get("pr_selectivity")
        return cls(
            material_id=str_field(value, "material_id"),
            role=str_field(value, "role"),
            phase=str_field(value, "phase"),
            composition=float_map(value.get("composition", {}), "composition"),
            pr_selectivity=None if pr_value is None else float_field(value, "pr_selectivity"),
        )


@dataclass(frozen=True, slots=True)
class MaterialStack:
    materials: tuple[MaterialModel, ...]

    @classmethod
    def from_mapping(cls, value: JsonMap) -> MaterialStack:
        return cls(
            materials=tuple(
                MaterialModel.from_mapping(as_mapping(item, "materials[]"))
                for item in as_sequence(value.get("materials", []), "materials")
            )
        )


@dataclass(frozen=True, slots=True)
class VolumeState:
    material_id: str
    phase: str
    initial_amorphous_index: float
    density_factor: float
    preexisting_damage: float
    implanted_inert_fraction: float
    rdf_order_features: dict[str, float]
    grain_or_orientation_id: str | None
    source_structure_id: str

    @classmethod
    def from_mapping(cls, value: JsonMap) -> VolumeState:
        return cls(
            material_id=str_field(value, "material_id"),
            phase=str_field(value, "phase"),
            initial_amorphous_index=float_field(value, "initial_amorphous_index"),
            density_factor=float_field(value, "density_factor"),
            preexisting_damage=float_field(value, "preexisting_damage"),
            implanted_inert_fraction=float_field(value, "implanted_inert_fraction"),
            rdf_order_features=float_map(value.get("rdf_order_features", {}), "rdf_order_features"),
            grain_or_orientation_id=optional_str(value, "grain_or_orientation_id"),
            source_structure_id=str_field(value, "source_structure_id"),
        )


@dataclass(frozen=True, slots=True)
class SurfaceState:
    material_id: str
    phase: str
    amorphous_index: float
    damage_dose: float
    roughness_rms_nm: float
    roughness_corr_length_nm: float
    implanted_inert_fraction: float
    local_fluence: float
    removed_depth_nm: float
    rdf_crystal_similarity: float
    rdf_amorphous_similarity: float
    coordination_defect_fraction: float
    active_layer_thickness_nm: float
    kernel_version: str

    @classmethod
    def from_mapping(cls, value: JsonMap) -> SurfaceState:
        return cls(
            material_id=str_field(value, "material_id"),
            phase=str_field(value, "phase"),
            amorphous_index=float_field(value, "amorphous_index"),
            damage_dose=float_field(value, "damage_dose"),
            roughness_rms_nm=float_field(value, "roughness_rms_nm"),
            roughness_corr_length_nm=float_field(value, "roughness_corr_length_nm"),
            implanted_inert_fraction=float_field(value, "implanted_inert_fraction"),
            local_fluence=float_field(value, "local_fluence"),
            removed_depth_nm=float_field(value, "removed_depth_nm"),
            rdf_crystal_similarity=float_field(value, "rdf_crystal_similarity"),
            rdf_amorphous_similarity=float_field(value, "rdf_amorphous_similarity"),
            coordination_defect_fraction=float_field(value, "coordination_defect_fraction"),
            active_layer_thickness_nm=float_field(value, "active_layer_thickness_nm"),
            kernel_version=str_field(value, "kernel_version"),
        )


@dataclass(frozen=True, slots=True)
class GeometrySource:
    kind: str
    path: str
    units: str

    @classmethod
    def from_mapping(cls, value: JsonMap) -> GeometrySource:
        return cls(kind=str_field(value, "kind"), path=str_field(value, "path"), units=str_field(value, "units"))


@dataclass(frozen=True, slots=True)
class SimulationScene:
    scene_id: str
    mode: str
    feature_type: str
    geometry_source: GeometrySource
    material_stack: MaterialStack
    volume_states: tuple[VolumeState, ...]
    surface_state: SurfaceState

    @classmethod
    def from_mapping(cls, value: JsonMap) -> SimulationScene:
        return cls(
            scene_id=str_field(value, "scene_id"),
            mode=str_field(value, "mode"),
            feature_type=str_field(value, "feature_type"),
            geometry_source=GeometrySource.from_mapping(as_mapping(value.get("geometry_source"), "geometry_source")),
            material_stack=MaterialStack.from_mapping(as_mapping(value.get("material_stack"), "material_stack")),
            volume_states=tuple(
                VolumeState.from_mapping(as_mapping(item, "volume_states[]"))
                for item in as_sequence(value.get("volume_states", []), "volume_states")
            ),
            surface_state=SurfaceState.from_mapping(as_mapping(value.get("surface_state"), "surface_state")),
        )
