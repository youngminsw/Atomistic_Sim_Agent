from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from sim_agent.compute import allowed_compute_hosts, require_allowed_host


UiMode = Literal["2d", "3d"]


@dataclass(frozen=True, slots=True)
class ControllerRunRequest:
    mode: UiMode
    geometry_path: str
    kernel_path: str
    events_path: str
    steps: int
    ions: int
    run_id: str
    compute_target: str
    iedf_ready: bool
    iadf_ready: bool
    output_dir: str | None = None


@dataclass(frozen=True, slots=True)
class ControllerValidation:
    can_run: bool
    missing_fields: tuple[str, ...]
    compute_target: str
    request: ControllerRunRequest


def validate_controller_request(request: ControllerRunRequest) -> ControllerValidation:
    target = require_allowed_host(request.compute_target)
    missing = _missing_fields(request)
    return ControllerValidation(
        can_run=not missing,
        missing_fields=missing,
        compute_target=target.host_alias,
        request=request,
    )


def build_offline_runner_command(request: ControllerRunRequest) -> tuple[str, ...]:
    source_flag = _source_flag(request.mode)
    return (
        "python",
        "02.Source_code/mss_agent/scripts/run_offline_simulation.py",
        source_flag,
        request.geometry_path,
        "--kernel",
        request.kernel_path,
        "--events",
        request.events_path,
        "--steps",
        str(request.steps),
        "--ions",
        str(request.ions),
        "--out",
        request.output_dir or f"02.Source_code/mss_agent/evidence/{request.run_id}",
        "--run-id",
        request.run_id,
    )


def controller_compute_targets() -> tuple[str, ...]:
    return allowed_compute_hosts()


def _missing_fields(request: ControllerRunRequest) -> tuple[str, ...]:
    missing: list[str] = []
    if not request.geometry_path:
        missing.append(_geometry_field(request.mode))
    if not request.kernel_path:
        missing.append("kernel")
    if not request.events_path:
        missing.append("events")
    if request.steps <= 0:
        missing.append("steps")
    if request.ions <= 0:
        missing.append("ions")
    if not request.iedf_ready:
        missing.append("iedf")
    if not request.iadf_ready:
        missing.append("iadf")
    return tuple(missing)


def _source_flag(mode: UiMode) -> str:
    match mode:
        case "2d":
            return "--image"
        case "3d":
            return "--scene"
        case unreachable:
            assert_never(unreachable)


def _geometry_field(mode: UiMode) -> str:
    match mode:
        case "2d":
            return "image"
        case "3d":
            return "scene"
        case unreachable:
            assert_never(unreachable)
