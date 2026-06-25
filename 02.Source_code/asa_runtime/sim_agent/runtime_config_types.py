from __future__ import annotations

from dataclasses import dataclass

from sim_agent.runtime_compute_config import ComputeResourceConfig
from sim_agent.runtime_graphdb_config import GraphDBRuntimeConfig


@dataclass(frozen=True, slots=True)
class ModelEndpointRuntimeConfig:
    provider: str
    model: str
    reasoning_effort: str
    base_url: str
    auth_mode: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class AgentModelRuntimeConfig:
    agent_id: str
    provider: str
    model: str
    reasoning_effort: str
    base_url: str
    auth_mode: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class ActiveModelProfileRuntimeConfig:
    name: str
    customized: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    workspace_root: str
    evidence_root: str
    team_mode_default: bool
    model_endpoint: ModelEndpointRuntimeConfig
    active_profile: ActiveModelProfileRuntimeConfig
    agent_model_overrides: tuple[AgentModelRuntimeConfig, ...]
    graphdb: GraphDBRuntimeConfig
    compute_resources: tuple[ComputeResourceConfig, ...]
