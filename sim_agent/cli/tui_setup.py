from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from sim_agent.runtime_config import (
    ComputeResourceConfig,
    load_runtime_config,
    runtime_config_path,
    save_runtime_config,
)

from .tui_parse import parse_options
from .tui_state import TuiState, append_event


def handle_setup(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    if not args:
        _write_setup_help(output_stream)
        return state
    command, *rest = args
    match command:
        case "runtime":
            return _handle_runtime_setup(tuple(rest), state, output_stream)
        case _:
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


def _write_setup_help(output_stream: TextIO) -> None:
    output_stream.write("setup_scope=runtime\n")
    _write_runtime_help(output_stream)


def _write_runtime_help(output_stream: TextIO) -> None:
    output_stream.write(
        "usage=/setup runtime --compute-resource <alias> "
        "--roles gpu,mdn,feature_scale --priority <n> "
        "--environment-name <env> [--ssh-target user@host --ssh-port 22]\n"
    )
    output_stream.write("usage=/setup runtime --remove-compute-resource <alias>\n")
    output_stream.write("usage=/setup runtime --list\n")
