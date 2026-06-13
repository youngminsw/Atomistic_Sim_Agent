from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClarificationPrompt:
    field: str
    question: str


@dataclass(frozen=True, slots=True)
class InputPlanningResult:
    request_id: str
    mode: str
    feature_type: str
    geometry_kind: str
    geometry_path: str
    geometry_units: str
    structure_description: str
    mask_material_id: str
    target_material_id: str
    ion_species: str
    missing_fields: tuple[str, ...]
    clarifications: tuple[ClarificationPrompt, ...]
    proposed_defaults: tuple[str, ...]
    model_training_required: bool
    training_reason: str
    trained_kernel_id: str

    @property
    def clarification_required(self) -> bool:
        return bool(self.missing_fields)
