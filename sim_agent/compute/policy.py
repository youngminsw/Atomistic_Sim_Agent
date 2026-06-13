from __future__ import annotations

from typing import Final

from .types import ComputePolicyError, ComputeTarget


REMOTE_HOST_PRIORITY: Final[tuple[str, ...]] = ("gpu-5090", "blackwell-rtxpro", "gpu-ada", "ws-gpu")
LOCAL_FALLBACK_HOST: Final[str] = "local"
COMPUTE_HOST_PRIORITY: Final[tuple[str, ...]] = REMOTE_HOST_PRIORITY + (LOCAL_FALLBACK_HOST,)


def allowed_compute_hosts() -> tuple[str, ...]:
    return COMPUTE_HOST_PRIORITY


def select_compute_target(
    available_hosts: tuple[str, ...],
    allow_local_fallback: bool = True,
) -> ComputeTarget:
    available = frozenset(available_hosts)
    for host_alias in REMOTE_HOST_PRIORITY:
        if host_alias in available:
            return ComputeTarget(host_alias=host_alias, remote=True, uses_local_fallback=False)
    if allow_local_fallback:
        return ComputeTarget(host_alias=LOCAL_FALLBACK_HOST, remote=False, uses_local_fallback=True)
    raise ComputePolicyError("no_allowed_compute_host_available")


def require_allowed_host(host_alias: str) -> ComputeTarget:
    if host_alias not in COMPUTE_HOST_PRIORITY:
        raise ComputePolicyError(f"host_not_allowed={host_alias}")
    return ComputeTarget(
        host_alias=host_alias,
        remote=host_alias != LOCAL_FALLBACK_HOST,
        uses_local_fallback=host_alias == LOCAL_FALLBACK_HOST,
    )
