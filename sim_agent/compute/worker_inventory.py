from __future__ import annotations

import os
from dataclasses import dataclass

from .policy import require_allowed_host
from .types import ComputePolicyError


@dataclass(frozen=True, slots=True)
class WorkerHostConfig:
    host_alias: str
    environment_name: str
    remote_user: str
    ssh_target: str | None
    ssh_port: int | None
    inventory_source: str


def resolve_worker_host(
    host_alias: str,
    environment_name: str,
    remote_user: str | None,
    ssh_target: str | None,
    ssh_port: int | None,
) -> WorkerHostConfig:
    target = require_allowed_host(host_alias)
    prefix = _env_prefix(host_alias)
    default = _default_config(host_alias)
    resolved_target = _first_text(ssh_target, os.environ.get(f"{prefix}_SSH_TARGET"), default.ssh_target)
    resolved_port = _first_int(ssh_port, os.environ.get(f"{prefix}_SSH_PORT"), default.ssh_port)
    resolved_user = _first_text(remote_user, os.environ.get(f"{prefix}_REMOTE_USER"), default.remote_user)
    resolved_env = _first_text(
        environment_name,
        os.environ.get(f"{prefix}_ENVIRONMENT_NAME"),
        default.environment_name,
    )
    source = _inventory_source(ssh_target, os.environ.get(f"{prefix}_SSH_TARGET"), default.ssh_target)
    if target.uses_local_fallback:
        return WorkerHostConfig(host_alias, resolved_env, resolved_user, None, None, source)
    return WorkerHostConfig(host_alias, resolved_env, resolved_user, resolved_target, resolved_port, source)


def require_remote_worker_host(
    host_alias: str,
    environment_name: str,
    remote_user: str | None,
    ssh_target: str | None,
    ssh_port: int | None,
) -> WorkerHostConfig:
    config = resolve_worker_host(host_alias, environment_name, remote_user, ssh_target, ssh_port)
    if config.ssh_target is None or config.ssh_port is None:
        raise ComputePolicyError(f"ssh_target_required_for_host={host_alias}")
    return config


def _default_config(host_alias: str) -> WorkerHostConfig:
    if host_alias == "gpu-5090":
        return WorkerHostConfig(
            host_alias=host_alias,
            environment_name="atomistic-sim-gpu",
            remote_user="swym",
            ssh_target="swym@10.24.12.85",
            ssh_port=55555,
            inventory_source="default",
        )
    return WorkerHostConfig(
        host_alias=host_alias,
        environment_name="atomistic-sim-gpu",
        remote_user="swym",
        ssh_target=None,
        ssh_port=None,
        inventory_source="default",
    )


def _env_prefix(host_alias: str) -> str:
    return "ATOMISTIC_SIM_" + host_alias.upper().replace("-", "_")


def _first_text(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    raise ComputePolicyError("worker_host_text_default_missing")


def _first_int(explicit: int | None, env_value: str | None, default: int | None) -> int | None:
    if explicit is not None:
        return explicit
    if env_value:
        return int(env_value)
    return default


def _inventory_source(explicit: str | None, env_value: str | None, default: str | None) -> str:
    if explicit:
        return "explicit"
    if env_value:
        return "environment"
    if default:
        return "default"
    return "missing"
