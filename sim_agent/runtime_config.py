from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.llm_endpoints.config import (
    DEFAULT_API_KEY_ENV_BY_PROVIDER,
    DEFAULT_AUTH_MODE_BY_PROVIDER,
    PRIMARY_MODEL,
    PRIMARY_REASONING,
    ModelProviderConfig,
)
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str


RUNTIME_CONFIG_ENV: Final = "ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"


@dataclass(frozen=True, slots=True)
class ComputeResourceConfig:
    host_alias: str
    roles: tuple[str, ...]
    priority: int
    environment_name: str
    remote_user: str
    ssh_target: str | None
    ssh_port: int | None
    local: bool = False


@dataclass(frozen=True, slots=True)
class ModelEndpointRuntimeConfig:
    provider: str
    model: str
    reasoning_effort: str
    base_url: str
    auth_mode: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class GraphDBRuntimeConfig:
    uri_env: str
    user_env: str
    password_env: str
    database: str


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    workspace_root: str
    evidence_root: str
    team_mode_default: bool
    model_endpoint: ModelEndpointRuntimeConfig
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
    config_path.write_text(json.dumps(runtime_config_payload(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def default_runtime_config() -> RuntimeConfig:
    source_root = Path(__file__).resolve().parents[1]
    return RuntimeConfig(
        workspace_root=str(source_root),
        evidence_root=str(source_root / "evidence" / "asa-runtime"),
        team_mode_default=True,
        model_endpoint=ModelEndpointRuntimeConfig(
            provider="openclaw",
            model=PRIMARY_MODEL,
            reasoning_effort=PRIMARY_REASONING,
            base_url="https://openclaw.local/v1",
            auth_mode=DEFAULT_AUTH_MODE_BY_PROVIDER["openclaw"],
            api_key_env=DEFAULT_API_KEY_ENV_BY_PROVIDER["openclaw"],
        ),
        graphdb=GraphDBRuntimeConfig(
            uri_env="ASA_NEO4J_URI",
            user_env="ASA_NEO4J_USER",
            password_env="ASA_NEO4J_PASSWORD",
            database="asa_sim_agent",
        ),
        compute_resources=_default_compute_resources(),
    )


def default_model_endpoint(config: RuntimeConfig | None = None) -> ModelProviderConfig:
    endpoint = (config or load_runtime_config()).model_endpoint
    return ModelProviderConfig.from_mapping(
        {
            "provider": endpoint.provider,
            "model": endpoint.model,
            "reasoning_effort": endpoint.reasoning_effort,
            "base_url": endpoint.base_url,
            "auth_mode": endpoint.auth_mode,
            "api_key_env": endpoint.api_key_env,
        }
    )


def runtime_config_payload(config: RuntimeConfig) -> JsonMap:
    return {
        "workspace_root": config.workspace_root,
        "evidence_root": config.evidence_root,
        "team_mode_default": config.team_mode_default,
        "model_endpoint": _model_payload(config.model_endpoint),
        "graphdb": {
            "uri_env": config.graphdb.uri_env,
            "user_env": config.graphdb.user_env,
            "password_env": config.graphdb.password_env,
            "database": config.graphdb.database,
        },
        "compute_resources": [_compute_payload(resource) for resource in config.compute_resources],
    }


def runtime_config_from_payload(payload: JsonMap) -> RuntimeConfig:
    default = default_runtime_config()
    endpoint = _model_from_payload(as_mapping(payload.get("model_endpoint", _model_payload(default.model_endpoint)), "model_endpoint"))
    graphdb = _graphdb_from_payload(as_mapping(payload.get("graphdb", runtime_config_payload(default)["graphdb"]), "graphdb"))
    resources_value = payload.get("compute_resources")
    resources = default.compute_resources if resources_value is None else _compute_resources_from_payload(resources_value)
    return RuntimeConfig(
        workspace_root=_optional_text(payload, "workspace_root", default.workspace_root),
        evidence_root=_optional_text(payload, "evidence_root", default.evidence_root),
        team_mode_default=_optional_bool(payload, "team_mode_default", default.team_mode_default),
        model_endpoint=endpoint,
        graphdb=graphdb,
        compute_resources=resources,
    )


def _default_compute_resources() -> tuple[ComputeResourceConfig, ...]:
    return (
        ComputeResourceConfig("gpu-5090", ("gpu", "mdn", "feature_scale"), 1, "atomistic-sim-gpu", "swym", "swym@10.24.12.85", 55555),
        ComputeResourceConfig("blackwell-rtxpro", ("gpu", "mdn", "feature_scale"), 2, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("gpu-ada", ("gpu", "mdn", "feature_scale", "md"), 3, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("4090-gpu-ws", ("gpu", "mdn", "feature_scale"), 4, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("ws-gpu", ("gpu", "mdn", "feature_scale"), 5, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("local-rtx4060", ("gpu", "mdn", "feature_scale"), 6, "atomistic-sim-local", "local", None, None, True),
        ComputeResourceConfig("orca-cpu", ("md", "lammps"), 7, "atomistic-sim-cpu", "swym", None, None),
        ComputeResourceConfig("ws-24core", ("md", "lammps"), 8, "atomistic-sim-cpu", "swym", None, None),
        ComputeResourceConfig("ws-96core", ("md", "lammps"), 9, "atomistic-sim-cpu", "swym", None, None),
    )


def _compute_resources_from_payload(value: object) -> tuple[ComputeResourceConfig, ...]:
    resources: list[ComputeResourceConfig] = []
    for item in as_sequence(value, "compute_resources"):
        resources.append(_compute_from_payload(as_mapping(item, "compute_resource")))
    return tuple(resources)


def _compute_from_payload(payload: JsonMap) -> ComputeResourceConfig:
    return ComputeResourceConfig(
        host_alias=as_str(payload.get("host_alias"), "host_alias"),
        roles=_roles(payload.get("roles", ())),
        priority=_optional_int(payload, "priority", 100),
        environment_name=_optional_text(payload, "environment_name", "atomistic-sim-gpu"),
        remote_user=_optional_text(payload, "remote_user", "swym"),
        ssh_target=_optional_text_or_none(payload, "ssh_target"),
        ssh_port=_optional_int_or_none(payload, "ssh_port"),
        local=_optional_bool(payload, "local", False),
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


def _graphdb_from_payload(payload: JsonMap) -> GraphDBRuntimeConfig:
    return GraphDBRuntimeConfig(
        uri_env=_optional_text(payload, "uri_env", "ASA_NEO4J_URI"),
        user_env=_optional_text(payload, "user_env", "ASA_NEO4J_USER"),
        password_env=_optional_text(payload, "password_env", "ASA_NEO4J_PASSWORD"),
        database=_optional_text(payload, "database", "asa_sim_agent"),
    )


def _compute_payload(resource: ComputeResourceConfig) -> JsonMap:
    return {
        "host_alias": resource.host_alias,
        "roles": list(resource.roles),
        "priority": resource.priority,
        "environment_name": resource.environment_name,
        "remote_user": resource.remote_user,
        "ssh_target": resource.ssh_target,
        "ssh_port": resource.ssh_port,
        "local": resource.local,
    }


def _model_payload(endpoint: ModelEndpointRuntimeConfig) -> JsonMap:
    return {
        "provider": endpoint.provider,
        "model": endpoint.model,
        "reasoning_effort": endpoint.reasoning_effort,
        "base_url": endpoint.base_url,
        "auth_mode": endpoint.auth_mode,
        "api_key_env": endpoint.api_key_env,
    }


def _roles(value: object) -> tuple[str, ...]:
    return tuple(as_str(item, "roles[]") for item in as_sequence(value, "roles"))


def _optional_text(payload: JsonMap, field: str, default: str) -> str:
    value = payload.get(field, default)
    return as_str(value, field)


def _optional_text_or_none(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    return as_str(value, field)


def _optional_bool(payload: JsonMap, field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be a boolean")


def _optional_int(payload: JsonMap, field: str, default: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be an integer")


def _optional_int_or_none(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be an integer")
