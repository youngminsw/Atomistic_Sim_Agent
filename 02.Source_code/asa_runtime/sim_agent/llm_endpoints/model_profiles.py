from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelProfileAssignment:
    agent_id: str
    provider: str
    model: str
    reasoning_effort: str

    @property
    def reference(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    label: str
    summary: str
    default: ModelProfileAssignment
    agents: tuple[ModelProfileAssignment, ...]


MODEL_PROFILES: Final[tuple[ModelProfile, ...]] = (
    ModelProfile(
        name="codex-eco",
        label="Codex Eco",
        summary="low-latency default with minimal execution lanes",
        default=ModelProfileAssignment("", "openai-codex", "gpt-5.5", "low"),
        agents=(
            ModelProfileAssignment("md_agent", "openai-codex", "gpt-5.5", "minimal"),
            ModelProfileAssignment("ml_mdn_agent", "openai-codex", "gpt-5.5", "low"),
            ModelProfileAssignment("feature_scale_agent", "openai-codex", "gpt-5.5", "low"),
            ModelProfileAssignment("research_graphdb_agent", "openai-codex", "gpt-5.5", "medium"),
            ModelProfileAssignment("qa_agent", "openai-codex", "gpt-5.5", "high"),
        ),
    ),
    ModelProfile(
        name="codex-medium",
        label="Codex Medium",
        summary="balanced orchestration with deeper critic and QA lanes",
        default=ModelProfileAssignment("", "openai-codex", "gpt-5.5", "medium"),
        agents=(
            ModelProfileAssignment("md_agent", "openai-codex", "gpt-5.5", "low"),
            ModelProfileAssignment("ml_mdn_agent", "openai-codex", "gpt-5.5", "medium"),
            ModelProfileAssignment("feature_scale_agent", "openai-codex", "gpt-5.5", "medium"),
            ModelProfileAssignment("research_graphdb_agent", "openai-codex", "gpt-5.5", "high"),
            ModelProfileAssignment("qa_agent", "openai-codex", "gpt-5.5", "xhigh"),
        ),
    ),
    ModelProfile(
        name="codex-pro",
        label="Codex Pro",
        summary="xhigh orchestrator with high-reasoning specialist lanes",
        default=ModelProfileAssignment("", "openai-codex", "gpt-5.5", "xhigh"),
        agents=(
            ModelProfileAssignment("md_agent", "openai-codex", "gpt-5.5", "medium"),
            ModelProfileAssignment("ml_mdn_agent", "openai-codex", "gpt-5.5", "high"),
            ModelProfileAssignment("feature_scale_agent", "openai-codex", "gpt-5.5", "high"),
            ModelProfileAssignment("research_graphdb_agent", "openai-codex", "gpt-5.5", "xhigh"),
            ModelProfileAssignment("qa_agent", "openai-codex", "gpt-5.5", "xhigh"),
        ),
    ),
)


def list_model_profiles() -> tuple[ModelProfile, ...]:
    return MODEL_PROFILES


def find_model_profile(name: str) -> ModelProfile | None:
    normalized = name.strip().lower()
    for profile in MODEL_PROFILES:
        if profile.name == normalized:
            return profile
    return None
