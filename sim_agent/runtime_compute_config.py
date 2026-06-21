from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap, as_sequence, as_str
from sim_agent.schemas.errors import SchemaValidationError


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


def default_compute_resources() -> tuple[ComputeResourceConfig, ...]:
    return (
        ComputeResourceConfig(
            "gpu-5090",
            ("gpu", "mdn", "feature_scale"),
            1,
            "atomistic-sim-gpu",
            "swym",
            "swym@10.24.12.85",
            55555,
        ),
        ComputeResourceConfig("blackwell-rtxpro", ("gpu", "mdn", "feature_scale"), 2, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("gpu-ada", ("gpu", "mdn", "feature_scale", "md"), 3, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("4090-gpu-ws", ("gpu", "mdn", "feature_scale"), 4, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("ws-gpu", ("gpu", "mdn", "feature_scale"), 5, "atomistic-sim-gpu", "swym", None, None),
        ComputeResourceConfig("local-rtx4060", ("gpu", "mdn", "feature_scale"), 6, "atomistic-sim-local", "local", None, None, True),
        ComputeResourceConfig("orca-cpu", ("md", "lammps"), 7, "atomistic-sim-cpu", "swym", None, None),
        ComputeResourceConfig("ws-24core", ("md", "lammps"), 8, "atomistic-sim-cpu", "swym", None, None),
        ComputeResourceConfig("ws-96core", ("md", "lammps"), 9, "atomistic-sim-cpu", "swym", None, None),
    )


def compute_from_payload(payload: JsonMap) -> ComputeResourceConfig:
    return ComputeResourceConfig(
        host_alias=as_str(payload.get("host_alias"), "host_alias"),
        roles=_roles_from_payload(payload),
        priority=_optional_int(payload, "priority", 100),
        environment_name=_optional_text(payload, "environment_name", "atomistic-sim-gpu"),
        remote_user=_optional_text(payload, "remote_user", "swym"),
        ssh_target=_optional_text_or_none(payload, "ssh_target"),
        ssh_port=_optional_int_or_none(payload, "ssh_port"),
        local=_optional_bool(payload, "local", False),
    )


def compute_payload(resource: ComputeResourceConfig) -> JsonMap:
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


def _roles_from_payload(payload: JsonMap) -> tuple[str, ...]:
    return tuple(as_str(item, "roles[]") for item in as_sequence(payload.get("roles", ()), "roles"))


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
    raise SchemaValidationError(f"{field} must be a boolean")


def _optional_int(payload: JsonMap, field: str, default: int) -> int:
    value = payload.get(field, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")


def _optional_int_or_none(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")
