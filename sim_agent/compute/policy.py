from __future__ import annotations

from sim_agent.runtime_config import ComputeResourceConfig, RuntimeConfig, load_runtime_config

from .types import ComputePolicyError, ComputeTarget


LOCAL_COMPAT_ALIAS = "local"


def allowed_compute_hosts(config: RuntimeConfig | None = None) -> tuple[str, ...]:
    return tuple(resource.host_alias for resource in _resources_by_priority(config))


def compute_resource_for_host(host_alias: str, config: RuntimeConfig | None = None) -> ComputeResourceConfig:
    normalized = _normalize_alias(host_alias)
    for resource in _resources_by_priority(config):
        if resource.host_alias == normalized:
            return resource
    raise ComputePolicyError(f"host_not_allowed={host_alias}")


def default_compute_resource(config: RuntimeConfig | None = None) -> ComputeResourceConfig:
    resources = _resources_by_priority(config)
    if not resources:
        raise ComputePolicyError("no_compute_resources_configured")
    return resources[0]


def select_compute_target(
    available_hosts: tuple[str, ...],
    allow_local_fallback: bool = True,
    config: RuntimeConfig | None = None,
) -> ComputeTarget:
    available = frozenset(available_hosts)
    for resource in _resources_by_priority(config):
        if resource.local:
            continue
        if "gpu" not in resource.roles:
            continue
        if resource.host_alias in available:
            return ComputeTarget(host_alias=resource.host_alias, remote=True, uses_local_fallback=False)
    if allow_local_fallback:
        local = _local_resource(config)
        return ComputeTarget(host_alias=local.host_alias, remote=False, uses_local_fallback=True)
    raise ComputePolicyError("no_allowed_compute_host_available")


def require_allowed_host(host_alias: str, config: RuntimeConfig | None = None) -> ComputeTarget:
    resource = compute_resource_for_host(host_alias, config)
    return ComputeTarget(
        host_alias=resource.host_alias,
        remote=not resource.local,
        uses_local_fallback=resource.local,
    )


def _resources_by_priority(config: RuntimeConfig | None = None) -> tuple[ComputeResourceConfig, ...]:
    resources = (config or load_runtime_config()).compute_resources
    return tuple(sorted(resources, key=lambda item: (item.priority, item.host_alias)))


def _local_resource(config: RuntimeConfig | None = None) -> ComputeResourceConfig:
    for resource in _resources_by_priority(config):
        if resource.local:
            return resource
    raise ComputePolicyError("local_fallback_not_configured")


def _normalize_alias(host_alias: str) -> str:
    if host_alias == LOCAL_COMPAT_ALIAS:
        return "local-rtx4060"
    return host_alias
