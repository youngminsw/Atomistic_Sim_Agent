from __future__ import annotations

import json
import shlex
from hashlib import sha256
from pathlib import Path, PurePosixPath

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, as_sequence, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .source_payload import SOURCE_PAYLOAD_ARCHIVE
from .types import (
    ComputePolicyError,
    RemoteExecutionChain,
    RemoteExecutionPlan,
    RemoteExecutionStage,
    WorkerBundle,
)


def load_worker_bundle(path: Path) -> WorkerBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComputePolicyError(f"worker_bundle_unreadable={path}") from exc
    except json.JSONDecodeError as exc:
        raise ComputePolicyError(f"worker_bundle_invalid_json={path}") from exc
    try:
        return _worker_from_mapping(as_mapping(payload, "worker_bundle"))
    except SchemaValidationError as exc:
        raise ComputePolicyError(str(exc)) from exc


def build_remote_execution_plan(
    bundle: WorkerBundle,
    ssh_target: str,
    ssh_port: int,
) -> RemoteExecutionPlan:
    if bundle.uses_local_fallback:
        raise ComputePolicyError("remote_plan_requires_remote_host")
    if ssh_port < 1 or ssh_port > 65535:
        raise ComputePolicyError(f"invalid_ssh_port={ssh_port}")
    return RemoteExecutionPlan(
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        local_setup_commands=_local_setup_commands(bundle),
        remote_setup_commands=_remote_setup_commands(bundle, ssh_target, ssh_port),
        upload_commands=_upload_commands(bundle, ssh_target, ssh_port),
        preflight_commands=_preflight_commands(bundle, ssh_target, ssh_port),
        execution_command=_ssh_command(ssh_target, ssh_port, bundle.command_line),
        download_commands=_download_commands(bundle, ssh_target, ssh_port),
    )


def build_remote_execution_chain(
    bundles: tuple[WorkerBundle, ...],
    ssh_target: str,
    ssh_port: int,
) -> RemoteExecutionChain:
    if len(bundles) < 2:
        raise ComputePolicyError("remote_execution_chain_requires_two_or_more_stages")
    stages = tuple(
        _remote_stage(index, bundle, ssh_target, ssh_port)
        for index, bundle in enumerate(bundles, start=1)
    )
    return RemoteExecutionChain(
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        stages=stages,
    )


def remote_execution_plan_payload(plan: RemoteExecutionPlan) -> JsonMap:
    return {
        "ssh_target": plan.ssh_target,
        "ssh_port": plan.ssh_port,
        "local_setup_commands": list(plan.local_setup_commands),
        "remote_setup_commands": list(plan.remote_setup_commands),
        "upload_commands": list(plan.upload_commands),
        "preflight_commands": list(plan.preflight_commands),
        "execution_command": plan.execution_command,
        "download_commands": list(plan.download_commands),
        "all_commands": list(plan.all_commands),
    }


def remote_execution_plan_manifest_payload(
    plan: RemoteExecutionPlan,
    source_root: Path,
    output_root: Path,
) -> JsonMap:
    payload = remote_execution_plan_payload(plan)
    return {
        "schema_version": 1,
        "kind": "remote_execution_plan",
        "created_by": "asa_runtime",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "plan_sha256": _commands_sha256(plan.all_commands),
        **payload,
    }


def remote_execution_chain_payload(chain: RemoteExecutionChain) -> JsonMap:
    return {
        "ssh_target": chain.ssh_target,
        "ssh_port": chain.ssh_port,
        "stage_count": len(chain.stages),
        "stages": [_stage_payload(stage) for stage in chain.stages],
        "all_commands": list(chain.all_commands),
    }


def _remote_stage(
    index: int,
    bundle: WorkerBundle,
    ssh_target: str,
    ssh_port: int,
) -> RemoteExecutionStage:
    plan = build_remote_execution_plan(bundle, ssh_target=ssh_target, ssh_port=ssh_port)
    return RemoteExecutionStage(
        stage_id=f"{index:02d}-{bundle.run_id}",
        run_id=bundle.run_id,
        plan=plan,
    )


def _stage_payload(stage: RemoteExecutionStage) -> JsonMap:
    plan_payload = remote_execution_plan_payload(stage.plan)
    return {
        "stage_id": stage.stage_id,
        "run_id": stage.run_id,
        "execution_command": stage.plan.execution_command,
        "commands": list(stage.plan.all_commands),
        "plan": plan_payload,
    }


def _worker_from_mapping(value: JsonMap) -> WorkerBundle:
    return WorkerBundle(
        host_alias=as_str(require(value, "host_alias"), "host_alias"),
        environment_name=as_str(require(value, "environment_name"), "environment_name"),
        run_id=as_str(require(value, "run_id"), "run_id"),
        remote_run_dir=_absolute_remote_path(
            as_str(require(value, "remote_run_dir"), "remote_run_dir")
        ),
        command_line=as_str(require(value, "command_line"), "command_line"),
        preflight_commands=_str_tuple(value, "preflight_commands"),
        capability_manifest_path=as_str(
            require(value, "capability_manifest_path"),
            "capability_manifest_path",
        ),
        capability_requirements=as_mapping(
            require(value, "capability_requirements"),
            "capability_requirements",
        ),
        input_paths=_str_tuple(value, "input_paths"),
        output_paths=_str_tuple(value, "output_paths"),
        transfer_plan=_str_tuple(value, "transfer_plan"),
        requires_cuda=as_bool(require(value, "requires_cuda"), "requires_cuda"),
        uses_local_fallback=as_bool(require(value, "uses_local_fallback"), "uses_local_fallback"),
    )


def _absolute_remote_path(raw: str) -> PurePosixPath:
    if not raw.startswith("/"):
        raise ComputePolicyError("remote_run_dir_must_be_absolute")
    return PurePosixPath(raw)


def _str_tuple(value: JsonMap, field: str) -> tuple[str, ...]:
    return tuple(as_str(item, field) for item in as_sequence(require(value, field), field))


def _local_setup_commands(bundle: WorkerBundle) -> tuple[str, ...]:
    dirs = _unique_relative_parent_dirs(bundle.output_paths)
    return tuple(f"mkdir -p {shlex.quote(path)}" for path in dirs)


def _remote_setup_commands(
    bundle: WorkerBundle,
    ssh_target: str,
    ssh_port: int,
) -> tuple[str, ...]:
    dirs = _unique_remote_parent_dirs(bundle)
    mkdir_command = "mkdir -p " + " ".join(shlex.quote(path) for path in dirs)
    return (_ssh_command(ssh_target, ssh_port, mkdir_command),)


def _upload_commands(bundle: WorkerBundle, ssh_target: str, ssh_port: int) -> tuple[str, ...]:
    return tuple(
        _rsync_command(local_path, f"{ssh_target}:{bundle.remote_run_dir / local_path}", ssh_port)
        for local_path in bundle.input_paths
    )


def _preflight_commands(
    bundle: WorkerBundle,
    ssh_target: str,
    ssh_port: int,
) -> tuple[str, ...]:
    preflight = tuple(
        _ssh_command(ssh_target, ssh_port, command)
        for command in bundle.preflight_commands
    )
    return _source_payload_extract_commands(bundle, ssh_target, ssh_port) + preflight


def _source_payload_extract_commands(
    bundle: WorkerBundle,
    ssh_target: str,
    ssh_port: int,
) -> tuple[str, ...]:
    if SOURCE_PAYLOAD_ARCHIVE not in bundle.input_paths:
        return ()
    command = f"cd {shlex.quote(str(bundle.remote_run_dir))} && tar -xzf {SOURCE_PAYLOAD_ARCHIVE}"
    return (_ssh_command(ssh_target, ssh_port, command),)


def _download_commands(bundle: WorkerBundle, ssh_target: str, ssh_port: int) -> tuple[str, ...]:
    output_paths = (bundle.capability_manifest_path,) + bundle.output_paths
    return tuple(
        _rsync_command(
            f"{ssh_target}:{bundle.remote_run_dir / output_path}",
            output_path,
            ssh_port,
        )
        for output_path in output_paths
    )


def _ssh_command(ssh_target: str, ssh_port: int, remote_command: str) -> str:
    return _join_shell(("ssh", "-p", str(ssh_port), ssh_target, remote_command))


def _rsync_command(source: str, destination: str, ssh_port: int) -> str:
    return _join_shell(("rsync", "-az", "-e", f"ssh -p {ssh_port}", source, destination))


def _join_shell(parts: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _commands_sha256(commands: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(commands),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _unique_remote_parent_dirs(bundle: WorkerBundle) -> tuple[str, ...]:
    dirs = [str(bundle.remote_run_dir)]
    dirs.extend(str((bundle.remote_run_dir / path).parent) for path in bundle.input_paths)
    dirs.extend(str((bundle.remote_run_dir / path).parent) for path in bundle.output_paths)
    dirs.append(str((bundle.remote_run_dir / bundle.capability_manifest_path).parent))
    return _unique_strings(tuple(dirs))


def _unique_relative_parent_dirs(paths: tuple[str, ...]) -> tuple[str, ...]:
    parents = tuple(str(Path(path).parent) for path in paths if str(Path(path).parent) != ".")
    return _unique_strings(parents)


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
