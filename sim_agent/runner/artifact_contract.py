from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunArtifactDescriptor:
    key: str
    filename: str


RUN_ARTIFACT_DESCRIPTORS: tuple[RunArtifactDescriptor, ...] = (
    RunArtifactDescriptor("manifest", "manifest.json"),
    RunArtifactDescriptor("profile_timeline", "timeline.json"),
    RunArtifactDescriptor("transport_field", "transport_field.json"),
    RunArtifactDescriptor("hit_history", "hit_history.json"),
    RunArtifactDescriptor("click_index", "click_index.json"),
    RunArtifactDescriptor("uncertainty_map", "uncertainty_map.json"),
    RunArtifactDescriptor("active_learning_plan", "active_learning_plan.json"),
    RunArtifactDescriptor("surrogate_model", "empirical_mdn_model.json"),
    RunArtifactDescriptor("surrogate_training_gate", "surrogate_training_gate_report.json"),
    RunArtifactDescriptor("surrogate_model_manifest", "surrogate_model_manifest.json"),
    RunArtifactDescriptor("qa_report", "qa_report.json"),
)


def run_artifact_types() -> list[str]:
    return [descriptor.key for descriptor in RUN_ARTIFACT_DESCRIPTORS]


def run_artifact_filename_map() -> dict[str, str]:
    return {descriptor.key: descriptor.filename for descriptor in RUN_ARTIFACT_DESCRIPTORS}
