from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from sim_agent.runtime_config import (
    ComputeResourceConfig,
    GraphDBRuntimeConfig,
    ModelEndpointRuntimeConfig,
    load_runtime_config,
    runtime_config_path,
    save_runtime_config,
)
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError

from .tui_model_endpoint import reasoning_effort_from_options
from .tui_parse import parse_options
from .tui_state import ModelSettings, TuiState, append_event, replace_model


def handle_setup(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    if not args:
        _write_setup_help(output_stream)
        return state
    command, *rest = args
    if command == "runtime":
        return _handle_runtime_setup(tuple(rest), state, output_stream)
    if command == "graphdb":
        return _handle_graphdb_setup(tuple(rest), state, output_stream)
    if command == "endpoint":
        return _handle_endpoint_setup(tuple(rest), state, output_stream)
    output_stream.write(f"setup_error=unknown_setup_scope:{command}\n")
    _write_setup_help(output_stream)
    return state


def _handle_runtime_setup(args: tuple[str, ...], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    config = load_runtime_config()
    resources = config.compute_resources
    if "remove_compute_resource" in parsed.options:
        alias = parsed.options["remove_compute_resource"]
        resources = tuple(resource for resource in resources if resource.host_alias != alias)
        output_stream.write(f"runtime_compute_resource_removed={alias}\n")
    elif "compute_resource" in parsed.options:
        resource = _resource_from_options(parsed.options, parsed.flags)
        resources = _upsert_resource(resources, resource)
        output_stream.write(f"runtime_compute_resource_saved={resource.host_alias}\n")
    elif parsed.flags and "list" in parsed.flags:
        _write_runtime_config(config.compute_resources, output_stream)
        return state
    else:
        _write_runtime_help(output_stream)
        return state

    path = save_runtime_config(replace(config, compute_resources=resources))
    append_event(state, "runtime_config_saved", str(path))
    output_stream.write(f"runtime_config_path={path}\n")
    output_stream.write(f"runtime_compute_resource_count={len(resources)}\n")
    return state


def _handle_graphdb_setup(args: tuple[str, ...], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    config = load_runtime_config()
    if "list" in parsed.flags:
        _write_graphdb_config(config.graphdb, output_stream)
        return state
    if not parsed.options:
        _write_graphdb_help(output_stream)
        return state

    graphdb = GraphDBRuntimeConfig(
        uri=parsed.options.get("uri", config.graphdb.uri),
        uri_env=parsed.options.get("uri_env", config.graphdb.uri_env),
        user_env=parsed.options.get("user_env", config.graphdb.user_env),
        password_env=parsed.options.get("password_env", config.graphdb.password_env),
        database=parsed.options.get("database", config.graphdb.database),
    )
    path = save_runtime_config(replace(config, graphdb=graphdb))
    append_event(state, "graphdb_config_saved", graphdb.database)
    output_stream.write("graphdb_config_saved=true\n")
    _write_graphdb_config(graphdb, output_stream)
    output_stream.write(f"runtime_config_path={path}\n")
    return state


def _handle_endpoint_setup(args: tuple[str, ...], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    config = load_runtime_config()
    if "list" in parsed.flags:
        _write_endpoint_config(config.model_endpoint, output_stream)
        return state
    if not parsed.options:
        _write_endpoint_help(output_stream)
        return state

    candidate = ModelEndpointRuntimeConfig(
        provider=parsed.options.get("provider", config.model_endpoint.provider),
        model=parsed.options.get("model", config.model_endpoint.model),
        reasoning_effort=reasoning_effort_from_options(parsed.options, config.model_endpoint.reasoning_effort),
        base_url=parsed.options.get("base_url", config.model_endpoint.base_url),
        auth_mode=parsed.options.get("auth_mode", config.model_endpoint.auth_mode),
        api_key_env=parsed.options.get("api_key_env", config.model_endpoint.api_key_env),
    )
    try:
        normalized = _validated_endpoint(candidate)
    except (ModelPolicyError, ProviderConfigPolicyError) as exc:
        output_stream.write(f"endpoint_config_error={exc}\n")
        append_event(state, "endpoint_config_blocked", str(exc))
        return state

    path = save_runtime_config(replace(config, model_endpoint=normalized))
    next_state = replace_model(state, _model_settings(normalized))
    append_event(next_state, "endpoint_config_saved", f"{normalized.provider}/{normalized.model}")
    output_stream.write("endpoint_config_saved=true\n")
    _write_endpoint_config(normalized, output_stream)
    output_stream.write(f"runtime_config_path={path}\n")
    return next_state


def _resource_from_options(options: dict[str, str], flags: tuple[str, ...]) -> ComputeResourceConfig:
    alias = options["compute_resource"]
    return ComputeResourceConfig(
        host_alias=alias,
        roles=_roles(options.get("roles", "gpu,mdn,feature_scale")),
        priority=_positive_int(options.get("priority", "100")),
        environment_name=options.get("environment_name", "atomistic-sim-gpu"),
        remote_user=options.get("remote_user", "swym"),
        ssh_target=_blank_as_none(options.get("ssh_target")),
        ssh_port=_optional_positive_int(options.get("ssh_port")),
        local="local" in flags,
    )


def _upsert_resource(
    resources: tuple[ComputeResourceConfig, ...],
    resource: ComputeResourceConfig,
) -> tuple[ComputeResourceConfig, ...]:
    kept = tuple(item for item in resources if item.host_alias != resource.host_alias)
    return (*kept, resource)


def _roles(value: str) -> tuple[str, ...]:
    roles = tuple(role.strip() for role in value.split(",") if role.strip())
    if not roles:
        return ("gpu",)
    return roles


def _positive_int(value: str) -> int:
    if value.isdecimal() and int(value) > 0:
        return int(value)
    return 100


def _optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    if value.isdecimal() and int(value) > 0:
        return int(value)
    return None


def _blank_as_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value


def _write_runtime_config(resources: tuple[ComputeResourceConfig, ...], output_stream: TextIO) -> None:
    output_stream.write(f"runtime_config_path={runtime_config_path()}\n")
    for resource in sorted(resources, key=lambda item: (item.priority, item.host_alias)):
        output_stream.write(
            "compute_resource="
            f"{resource.host_alias} roles={','.join(resource.roles)} "
            f"priority={resource.priority} env={resource.environment_name}\n"
        )


def _validated_endpoint(endpoint: ModelEndpointRuntimeConfig) -> ModelEndpointRuntimeConfig:
    normalized = ModelProviderConfig.from_mapping(
        {
            "provider": endpoint.provider,
            "model": endpoint.model,
            "reasoning_effort": endpoint.reasoning_effort,
            "base_url": endpoint.base_url,
            "auth_mode": endpoint.auth_mode,
            "api_key_env": endpoint.api_key_env,
        }
    )
    return ModelEndpointRuntimeConfig(
        provider=normalized.provider,
        model=normalized.model,
        reasoning_effort=normalized.reasoning_effort,
        base_url=normalized.base_url,
        auth_mode=normalized.auth_mode,
        api_key_env=normalized.api_key_env,
    )


def _model_settings(endpoint: ModelEndpointRuntimeConfig) -> ModelSettings:
    return ModelSettings(
        provider=endpoint.provider,
        name=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )


def _write_graphdb_config(graphdb: GraphDBRuntimeConfig, output_stream: TextIO) -> None:
    output_stream.write(f"graphdb_uri={graphdb.uri}\n")
    output_stream.write(f"graphdb_uri_env={graphdb.uri_env}\n")
    output_stream.write(f"graphdb_user_env={graphdb.user_env}\n")
    output_stream.write(f"graphdb_password_env={graphdb.password_env}\n")
    output_stream.write(f"graphdb_database={graphdb.database}\n")


def _write_endpoint_config(endpoint: ModelEndpointRuntimeConfig, output_stream: TextIO) -> None:
    output_stream.write(
        f"provider={endpoint.provider} model={endpoint.model} "
        f"reasoning_effort={endpoint.reasoning_effort}\n"
    )
    output_stream.write(f"base_url={endpoint.base_url} auth_mode={endpoint.auth_mode}\n")
    output_stream.write(f"api_key_env={endpoint.api_key_env}\n")


def _write_setup_help(output_stream: TextIO) -> None:
    output_stream.write("setup_scope=runtime\n")
    output_stream.write("setup_scope=graphdb\n")
    output_stream.write("setup_scope=endpoint\n")
    _write_runtime_help(output_stream)
    _write_graphdb_help(output_stream)
    _write_endpoint_help(output_stream)


def _write_runtime_help(output_stream: TextIO) -> None:
    output_stream.write(
        "usage=/setup runtime --compute-resource <alias> "
        "--roles gpu,mdn,feature_scale --priority <n> "
        "--environment-name <env> [--ssh-target user@host --ssh-port 22]\n"
    )
    output_stream.write("usage=/setup runtime --remove-compute-resource <alias>\n")
    output_stream.write("usage=/setup runtime --list\n")


def _write_graphdb_help(output_stream: TextIO) -> None:
    output_stream.write(
        "usage=/setup graphdb --uri-env NEO4J_URI --user-env NEO4J_USERNAME "
        "--password-env NEO4J_PASSWORD --database <db> [--uri bolt://server:7687]\n"
    )
    output_stream.write("usage=/setup graphdb --list\n")


def _write_endpoint_help(output_stream: TextIO) -> None:
    output_stream.write(
        "usage=/setup endpoint --provider <id> --model <model> --thinking-level high "
        "--base-url https://gateway/v1 --auth-mode oauth|gateway|api_key|none "
        "--api-key-env <ENV>\n"
    )
    output_stream.write("usage=/setup endpoint --list\n")
