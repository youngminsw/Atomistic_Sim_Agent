from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import PurePosixPath


class ComputePolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ComputeTarget:
    host_alias: str
    remote: bool
    uses_local_fallback: bool


@dataclass(frozen=True, slots=True)
class JobBundleSpec:
    job_id: str
    environment_name: str
    command: tuple[str, ...]
    input_paths: tuple[str, ...]
    output_paths: tuple[str, ...]
    requires_cuda: bool


@dataclass(frozen=True, slots=True)
class WorkerBundle:
    host_alias: str
    environment_name: str
    run_id: str
    remote_run_dir: PurePosixPath
    command_line: str
    preflight_commands: tuple[str, ...]
    capability_manifest_path: str
    capability_requirements: Mapping[str, object]
    input_paths: tuple[str, ...]
    output_paths: tuple[str, ...]
    transfer_plan: tuple[str, ...]
    requires_cuda: bool
    uses_local_fallback: bool


@dataclass(frozen=True, slots=True)
class RemoteExecutionPlan:
    ssh_target: str
    ssh_port: int
    local_setup_commands: tuple[str, ...]
    remote_setup_commands: tuple[str, ...]
    upload_commands: tuple[str, ...]
    preflight_commands: tuple[str, ...]
    execution_command: str
    download_commands: tuple[str, ...]

    @property
    def all_commands(self) -> tuple[str, ...]:
        return (
            self.local_setup_commands
            + self.remote_setup_commands
            + self.upload_commands
            + self.preflight_commands
            + (self.execution_command,)
            + self.download_commands
        )


@dataclass(frozen=True, slots=True)
class RemoteExecutionStage:
    stage_id: str
    run_id: str
    plan: RemoteExecutionPlan


@dataclass(frozen=True, slots=True)
class RemoteExecutionChain:
    ssh_target: str
    ssh_port: int
    stages: tuple[RemoteExecutionStage, ...]

    @property
    def all_commands(self) -> tuple[str, ...]:
        commands: tuple[str, ...] = ()
        for stage in self.stages:
            commands += stage.plan.all_commands
        return commands
