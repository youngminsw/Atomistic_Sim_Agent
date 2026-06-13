from __future__ import annotations

from dataclasses import dataclass

from ._parse import JsonMap, as_mapping, require, str_field
from .distributions import FluxSchedule, IonAngularDistribution, IonEnergyDistribution, SpeciesMix
from .errors import SchemaValidationError
from .state import SimulationScene
from sim_agent.llm_endpoints.config import ModelProviderConfig


@dataclass(frozen=True, slots=True)
class LLMEndpointConfig:
    provider: str
    model: str
    reasoning_effort: str
    base_url: str
    auth_mode: str

    @classmethod
    def from_mapping(cls, value: JsonMap) -> LLMEndpointConfig:
        provider_config = ModelProviderConfig.from_mapping(value)
        return cls(
            provider=provider_config.provider,
            model=provider_config.model,
            reasoning_effort=provider_config.reasoning_effort,
            base_url=provider_config.base_url,
            auth_mode=provider_config.auth_mode,
        )


@dataclass(frozen=True, slots=True)
class EtchRecipe:
    ion_species: str
    ion_energy_distribution: IonEnergyDistribution
    ion_angular_distribution: IonAngularDistribution
    flux_schedule: FluxSchedule
    species_mix: SpeciesMix

    @classmethod
    def from_mapping(cls, value: JsonMap) -> EtchRecipe:
        if "ion_energy_distribution" not in value:
            raise SchemaValidationError("IonEnergyDistribution required")
        return cls(
            ion_species=str_field(value, "ion_species"),
            ion_energy_distribution=IonEnergyDistribution.from_mapping(
                as_mapping(value.get("ion_energy_distribution"), "ion_energy_distribution")
            ),
            ion_angular_distribution=IonAngularDistribution.from_mapping(
                as_mapping(value.get("ion_angular_distribution"), "ion_angular_distribution")
            ),
            flux_schedule=FluxSchedule.from_mapping(as_mapping(value.get("flux_schedule"), "flux_schedule")),
            species_mix=SpeciesMix.from_sequence(require(value, "species_mix")),
        )


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    request_id: str
    llm_endpoint: LLMEndpointConfig
    scene: SimulationScene
    recipe: EtchRecipe

    @classmethod
    def from_mapping(cls, value: object) -> SimulationRequest:
        mapping = as_mapping(value, "simulation_request")
        return cls(
            request_id=str_field(mapping, "request_id"),
            llm_endpoint=LLMEndpointConfig.from_mapping(as_mapping(mapping.get("llm_endpoint"), "llm_endpoint")),
            scene=SimulationScene.from_mapping(as_mapping(mapping.get("scene"), "scene")),
            recipe=EtchRecipe.from_mapping(as_mapping(mapping.get("recipe"), "recipe")),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class RunArtifact:
    artifact_id: str
    path: str
    artifact_type: str


@dataclass(frozen=True, slots=True)
class ClickDiagnostic:
    position_id: str
    material_id: str
    profile_step: int
