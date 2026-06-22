from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .types import ClarificationPrompt, InputPlanningResult


@dataclass(frozen=True, slots=True)
class ExpertKernel:
    ion_species: str
    target_material_id: str
    kernel_id: str


KNOWN_EXPERTS: Final = (ExpertKernel("Ar", "Si", "Ar_on_Si__physical_v001"),)
QUESTION_TEXT: Final = (
    ("geometry", "Provide a 2D image or 3D mesh/layout/CAD geometry source before simulation."),
    ("material", "Provide PR mask material and target material composition before simulation."),
    ("phase", "Provide crystal or amorphous initial substrate phase and surface state before simulation."),
    ("iedf", "Provide an IonEnergyDistribution; the planner will not invent a fixed ion energy."),
    ("iadf", "Provide an IonAngularDistribution with polar and azimuthal angle ranges."),
    ("flux", "Provide a flux schedule or fluence target before time evolution."),
    ("ion_species", "Provide the incident ion species before choosing or training a surrogate expert."),
)


def plan_simulation_input(payload: JsonMap) -> InputPlanningResult:
    scene = _mapping_or_empty(payload.get("scene"))
    recipe = _mapping_or_empty(payload.get("recipe"))
    geometry = _mapping_or_empty(scene.get("geometry_source"))
    material_stack = _material_stack(payload, scene)
    ion_species = _text(recipe.get("ion_species"), _text(payload.get("ion_species"), ""))
    target_material_id = _target_material(payload, material_stack)
    expert = _expert_for(ion_species, target_material_id)
    missing_fields = _missing_fields(scene, recipe, geometry, material_stack, ion_species)
    training_required = bool(ion_species and target_material_id and expert.kernel_id == "")
    training_reason = (
        f"no_trained_expert_for_{ion_species}_on_{target_material_id}" if training_required else ""
    )

    return InputPlanningResult(
        request_id=_text(payload.get("request_id"), "anonymous"),
        mode=_text(scene.get("mode"), _text(payload.get("mode"), "unspecified")),
        feature_type=_text(scene.get("feature_type"), _text(payload.get("feature_type"), "unspecified")),
        geometry_kind=_text(geometry.get("kind"), "unspecified"),
        geometry_path=_text(geometry.get("path"), ""),
        geometry_units=_text(geometry.get("units"), ""),
        structure_description=_structure_description(payload, scene),
        mask_material_id=_mask_material(payload, material_stack),
        target_material_id=target_material_id,
        ion_species=ion_species,
        missing_fields=missing_fields,
        clarifications=tuple(ClarificationPrompt(field, _question_for(field)) for field in missing_fields),
        proposed_defaults=(),
        model_training_required=training_required,
        training_reason=training_reason,
        trained_kernel_id=expert.kernel_id,
    )


def _missing_fields(
    scene: JsonMap,
    recipe: JsonMap,
    geometry: JsonMap,
    material_stack: JsonMap,
    ion_species: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not geometry.get("path"):
        missing.append("geometry")
    if not _target_material({}, material_stack):
        missing.append("material")
    if not _has_phase(scene, material_stack):
        missing.append("phase")
    if not ion_species:
        missing.append("ion_species")
    if "ion_energy_distribution" not in recipe:
        missing.append("iedf")
    if "ion_angular_distribution" not in recipe:
        missing.append("iadf")
    if "flux_schedule" not in recipe:
        missing.append("flux")
    return tuple(_dedupe(missing))


def _material_stack(payload: JsonMap, scene: JsonMap) -> JsonMap:
    scene_stack = scene.get("material_stack")
    if isinstance(scene_stack, Mapping):
        return scene_stack
    payload_stack = payload.get("material_stack")
    if isinstance(payload_stack, Mapping):
        return payload_stack
    return {}


def _target_material(payload: JsonMap, material_stack: JsonMap) -> str:
    direct = _text(payload.get("target_material"), "")
    if direct:
        return direct
    shorthand = _text(material_stack.get("target"), "")
    if shorthand:
        return shorthand
    return _material_by_role(material_stack, "target")


def _mask_material(payload: JsonMap, material_stack: JsonMap) -> str:
    direct = _text(payload.get("mask_material"), "")
    if direct:
        return direct
    shorthand = _text(material_stack.get("mask"), "")
    if shorthand:
        return shorthand
    return _material_by_role(material_stack, "mask")


def _material_by_role(material_stack: JsonMap, role: str) -> str:
    materials = material_stack.get("materials")
    if not isinstance(materials, Sequence) or isinstance(materials, str):
        return ""
    for item in materials:
        if isinstance(item, Mapping) and item.get("role") == role:
            return _text(item.get("material_id"), "")
    return ""


def _has_phase(scene: JsonMap, material_stack: JsonMap) -> bool:
    if _text(scene.get("initial_phase"), ""):
        return True
    if "surface_state" in scene or "volume_states" in scene:
        return True
    materials = material_stack.get("materials")
    if not isinstance(materials, Sequence) or isinstance(materials, str):
        return False
    return any(isinstance(item, Mapping) and _text(item.get("phase"), "") for item in materials)


def _structure_description(payload: JsonMap, scene: JsonMap) -> str:
    scene_description = _text(scene.get("structure_description"), "")
    if scene_description:
        return scene_description
    return _text(payload.get("scene_description"), _text(payload.get("structure_description"), ""))


def _expert_for(ion_species: str, target_material_id: str) -> ExpertKernel:
    for expert in KNOWN_EXPERTS:
        if expert.ion_species == ion_species and expert.target_material_id == target_material_id:
            return expert
    return ExpertKernel(ion_species, target_material_id, "")


def _question_for(field: str) -> str:
    for key, question in QUESTION_TEXT:
        if key == field:
            return question
    return f"Provide {field} before simulation."


def _mapping_or_empty(value) -> JsonMap:
    if isinstance(value, Mapping):
        return value
    return {}


def _text(value, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)
