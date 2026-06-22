from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.llm_endpoints.config import (
    DEFAULT_API_KEY_ENV_BY_PROVIDER,
    DEFAULT_AUTH_MODE_BY_PROVIDER,
    ModelUseCase,
    PRIMARY_REASONING,
    ModelProviderConfig,
)
from sim_agent.llm_endpoints.model_catalog import find_model_catalog_entry
from sim_agent.llm_endpoints.model_profiles import ModelProfileAssignment, find_model_profile
from sim_agent.runtime_compute_config import (
    ComputeResourceConfig,
    compute_from_payload,
    compute_payload,
    default_compute_resources,
)
from sim_agent.runtime_graphdb_config import (
    GraphDBRuntimeConfig,
    default_graphdb_config,
    graphdb_from_payload,
    graphdb_payload,
)
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str
from sim_agent.schemas.errors import SchemaValidationError


RUNTIME_CONFIG_ENV: Final = "ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"


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


def runtime_config_path() -> Path:
    configured = os.environ.get(RUNTIME_CONFIG_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".atomistic-sim-agent" / "runtime-config.json"


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    config_path = path or runtime_config_path()
    if not config_path.exists():
        return default_runtime_config()
    payload = as_mapping(json.loads(config_path.read_text(encoding="utf-8")), "runtime_config")
    return runtime_config_from_payload(payload)


def save_runtime_config(config: RuntimeConfig, path: Path | None = None) -> Path:
    config_path = path or runtime_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(runtime_config_payload(config), indent=2, sort_keys=True) + "\n"
    config_path.write_text(payload, encoding="utf-8")
    return config_path


def default_runtime_config() -> RuntimeConfig:
    source_root = Path(__file__).resolve().parents[1]
    return RuntimeConfig(
        workspace_root=str(source_root),
        evidence_root=str(source_root / "evidence" / "asa-runtime"),
        team_mode_default=True,
        model_endpoint=ModelEndpointRuntimeConfig(
            provider="openai-codex",
            model="gpt-5-codex",
            reasoning_effort=PRIMARY_REASONING,
            base_url="https://model-gateway.local/v1",
            auth_mode=DEFAULT_AUTH_MODE_BY_PROVIDER["openai-codex"],
            api_key_env=DEFAULT_API_KEY_ENV_BY_PROVIDER["openai-codex"],
        ),
        active_profile=ActiveModelProfileRuntimeConfig(name="", customized=False),
        agent_model_overrides=(),
        graphdb=default_graphdb_config(),
        compute_resources=default_compute_resources(),
    )


def default_model_endpoint(config: RuntimeConfig | None = None) -> ModelProviderConfig:
    endpoint = (config or load_runtime_config()).model_endpoint
    payload = _model_payload(endpoint)
    payload["use_case"] = ModelUseCase.LOW_RISK_SUMMARIZATION.value
    return ModelProviderConfig.from_mapping(payload)


def runtime_config_payload(config: RuntimeConfig) -> JsonMap:
    return {
        "workspace_root": config.workspace_root,
        "evidence_root": config.evidence_root,
        "team_mode_default": config.team_mode_default,
        "model_endpoint": _model_payload(config.model_endpoint),
        "active_profile": _active_profile_payload(config.active_profile),
        "agent_model_overrides": [_agent_model_payload(item) for item in config.agent_model_overrides],
        "graphdb": graphdb_payload(config.graphdb),
        "compute_resources": [compute_payload(resource) for resource in config.compute_resources],
    }


def runtime_config_from_payload(payload: JsonMap) -> RuntimeConfig:
    default = default_runtime_config()
    endpoint = _model_from_payload(
        as_mapping(payload.get("model_endpoint", _model_payload(default.model_endpoint)), "model_endpoint")
    )
    graphdb_default = graphdb_payload(default.graphdb)
    resources = default.compute_resources
    if payload.get("compute_resources") is not None:
        resources = tuple(
            compute_from_payload(as_mapping(item, "compute_resource"))
            for item in as_sequence(payload["compute_resources"], "compute_resources")
        )
    return RuntimeConfig(
        workspace_root=_optional_text(payload, "workspace_root", default.workspace_root),
        evidence_root=_optional_text(payload, "evidence_root", default.evidence_root),
        team_mode_default=_optional_bool(payload, "team_mode_default", default.team_mode_default),
        model_endpoint=endpoint,
        active_profile=_active_profile_from_payload(payload, default.active_profile),
        agent_model_overrides=_agent_models_from_payload(payload),
        graphdb=graphdb_from_payload(as_mapping(payload.get("graphdb", graphdb_default), "graphdb")),
        compute_resources=resources,
    )


def active_profile_status(config: RuntimeConfig) -> ActiveModelProfileRuntimeConfig:
    active = config.active_profile
    if not active.name:
        return active
    profile = find_model_profile(active.name)
    if profile is None:
        return ActiveModelProfileRuntimeConfig(name=active.name, customized=True)
    if active.customized or not _config_matches_profile(config, profile.default, profile.agents):
        return ActiveModelProfileRuntimeConfig(name=active.name, customized=True)
    return active


def mark_active_profile_customized(config: RuntimeConfig) -> RuntimeConfig:
    if not config.active_profile.name:
        return config
    return RuntimeConfig(
        workspace_root=config.workspace_root,
        evidence_root=config.evidence_root,
        team_mode_default=config.team_mode_default,
        model_endpoint=config.model_endpoint,
        active_profile=ActiveModelProfileRuntimeConfig(name=config.active_profile.name, customized=True),
        agent_model_overrides=config.agent_model_overrides,
        graphdb=config.graphdb,
        compute_resources=config.compute_resources,
    )


def agent_model_override_by_id(config: RuntimeConfig) -> dict[str, AgentModelRuntimeConfig]:
    return {override.agent_id: override for override in config.agent_model_overrides}


def upsert_agent_model_override(
    overrides: tuple[AgentModelRuntimeConfig, ...],
    override: AgentModelRuntimeConfig,
) -> tuple[AgentModelRuntimeConfig, ...]:
    kept = tuple(item for item in overrides if item.agent_id != override.agent_id)
    return (*kept, override)


def remove_agent_model_override(
    overrides: tuple[AgentModelRuntimeConfig, ...],
    agent_id: str,
) -> tuple[AgentModelRuntimeConfig, ...]:
    return tuple(item for item in overrides if item.agent_id != agent_id)


def _agent_models_from_payload(payload: JsonMap) -> tuple[AgentModelRuntimeConfig, ...]:
    return tuple(
        _agent_model_from_payload(as_mapping(item, "agent_model_override"))
        for item in as_sequence(payload.get("agent_model_overrides", ()), "agent_model_overrides")
    )


def _active_profile_from_payload(
    payload: JsonMap,
    default: ActiveModelProfileRuntimeConfig,
) -> ActiveModelProfileRuntimeConfig:
    value = payload.get("active_profile")
    if value is None:
        return default
    if isinstance(value, str):
        return ActiveModelProfileRuntimeConfig(name=value, customized=False)
    mapping = as_mapping(value, "active_profile")
    return ActiveModelProfileRuntimeConfig(
        name=_optional_profile_name(mapping, default.name),
        customized=_optional_bool(mapping, "customized", default.customized),
    )


def _model_from_payload(payload: JsonMap) -> ModelEndpointRuntimeConfig:
    return ModelEndpointRuntimeConfig(
        provider=as_str(payload.get("provider"), "provider"),
        model=as_str(payload.get("model"), "model"),
        reasoning_effort=_optional_text(payload, "reasoning_effort", PRIMARY_REASONING),
        base_url=as_str(payload.get("base_url"), "base_url"),
        auth_mode=as_str(payload.get("auth_mode"), "auth_mode"),
        api_key_env=as_str(payload.get("api_key_env"), "api_key_env"),
    )


def _agent_model_from_payload(payload: JsonMap) -> AgentModelRuntimeConfig:
    endpoint = _model_from_payload(payload)
    return AgentModelRuntimeConfig(
        agent_id=as_str(payload.get("agent_id"), "agent_id"),
        provider=endpoint.provider,
        model=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )


def _model_payload(endpoint: ModelEndpointRuntimeConfig | AgentModelRuntimeConfig) -> JsonMap:
    return {
        "provider": endpoint.provider,
        "model": endpoint.model,
        "reasoning_effort": endpoint.reasoning_effort,
        "base_url": endpoint.base_url,
        "auth_mode": endpoint.auth_mode,
        "api_key_env": endpoint.api_key_env,
    }


def _agent_model_payload(override: AgentModelRuntimeConfig) -> JsonMap:
    payload = _model_payload(override)
    payload["agent_id"] = override.agent_id
    return payload


def _active_profile_payload(active_profile: ActiveModelProfileRuntimeConfig) -> JsonMap:
    return {
        "name": active_profile.name,
        "customized": active_profile.customized,
    }


def _config_matches_profile(
    config: RuntimeConfig,
    default: ModelProfileAssignment,
    agents: tuple[ModelProfileAssignment, ...],
) -> bool:
    if _model_payload(config.model_endpoint) != _model_payload(_endpoint_from_profile_assignment(default)):
        return False
    expected = {
        assignment.agent_id: _model_payload(_endpoint_from_profile_assignment(assignment))
        for assignment in agents
    }
    actual = {
        override.agent_id: _model_payload(override)
        for override in config.agent_model_overrides
    }
    return actual == expected


def _endpoint_from_profile_assignment(assignment: ModelProfileAssignment) -> ModelEndpointRuntimeConfig:
    entry = find_model_catalog_entry(assignment.reference)
    if entry is None:
        return ModelEndpointRuntimeConfig(
            provider=assignment.provider,
            model=assignment.model,
            reasoning_effort=assignment.reasoning_effort,
            base_url="https://model-gateway.local/v1",
            auth_mode="oauth",
            api_key_env="MODEL_GATEWAY_TOKEN",
        )
    return ModelEndpointRuntimeConfig(
        provider=entry.provider,
        model=entry.model,
        reasoning_effort=assignment.reasoning_effort,
        base_url=entry.base_url,
        auth_mode=entry.auth_mode,
        api_key_env=entry.api_key_env,
    )


def _optional_text(payload: JsonMap, field: str, default: str) -> str:
    value = payload.get(field, default)
    return as_str(value, field)


def _optional_profile_name(payload: JsonMap, default: str) -> str:
    value = payload.get("name", default)
    if isinstance(value, str):
        return value.strip().lower()
    raise SchemaValidationError("active_profile.name must be a string")


def _optional_bool(payload: JsonMap, field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be a boolean")
