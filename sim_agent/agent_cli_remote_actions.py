from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sim_agent.compute import (
    ComputePolicyError,
    prepare_remote_capability_probe,
    run_remote_execution_plan,
    run_remote_capability_probe,
    run_remote_chain,
    write_remote_execution_plan_result,
    write_remote_capability_probe_result,
    write_remote_chain_result,
)
from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class AgentCliRemoteActionConfig:
    source_root: Path
    host: str
    environment_name: str
    remote_user: str
    ssh_target: str | None
    ssh_port: int | None
    remote_run_timeout_s: float | None
    run_amorphous_structure_prep: bool
    run_remote_capability_probe: bool
    run_remote_chain: bool
    surrogate_training_gate_report: str | None


@dataclass(frozen=True, slots=True)
class AgentCliRemoteActionResult:
    exit_code: int
    amorphous_prep_result_path: Path | None
    capability_result_path: Path | None
    chain_result_path: Path | None
    surrogate_gate_path: Path | None


def run_requested_remote_actions(
    config: AgentCliRemoteActionConfig,
    response: JsonMap,
    output_dir: Path,
) -> AgentCliRemoteActionResult:
    exit_code = 0
    amorphous_prep_result_path: Path | None = None
    capability_result_path: Path | None = None
    chain_result_path: Path | None = None
    surrogate_gate_path = _optional_path(config.surrogate_training_gate_report)
    if config.run_amorphous_structure_prep:
        amorphous_prep_result_path, prep_ok = _run_amorphous_structure_prep(
            response,
            output_dir,
            config.remote_run_timeout_s,
        )
        if not prep_ok:
            exit_code = 1
    if config.run_remote_capability_probe:
        capability_result_path, probe_ok = _run_remote_capability_probe(config, output_dir)
        if not probe_ok:
            exit_code = 1
    if config.run_remote_chain:
        chain_result_path, chain_ok = _run_remote_chain(
            response,
            output_dir,
            config.remote_run_timeout_s,
        )
        if not chain_ok:
            exit_code = 1
    if _surrogate_gate_blocks_feature_scale(surrogate_gate_path):
        exit_code = 1
    return AgentCliRemoteActionResult(
        exit_code=exit_code,
        amorphous_prep_result_path=amorphous_prep_result_path,
        capability_result_path=capability_result_path,
        chain_result_path=chain_result_path,
        surrogate_gate_path=surrogate_gate_path,
    )


def _optional_path(raw: str | None) -> Path | None:
    return Path(raw) if raw else None


def _surrogate_gate_blocks_feature_scale(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return not (isinstance(payload, dict) and payload.get("accepted") is True)


def _run_remote_capability_probe(
    config: AgentCliRemoteActionConfig,
    output_dir: Path,
) -> tuple[Path, bool]:
    bundle = prepare_remote_capability_probe(
        source_root=config.source_root,
        output_dir=output_dir,
        host_alias=config.host,
        environment_name=config.environment_name,
        remote_user=config.remote_user,
        ssh_target=config.ssh_target,
        ssh_port=config.ssh_port,
        requires_cuda=True,
        requires_lammps=True,
        required_lammps_packages=("MANYBODY",),
    )
    result = run_remote_capability_probe(bundle.manifest_path, config.remote_run_timeout_s)
    result_path = output_dir / "remote_capability_probe_result.json"
    write_remote_capability_probe_result(result_path, result)
    print(f"remote_capability_probe_result_path={result_path}")
    print(f"remote_capability_probe_status={result.payload['probe_status']}")
    return (result_path, result.ok)


def _run_remote_chain(
    response: JsonMap,
    output_dir: Path,
    timeout_s: float | None,
) -> tuple[Path, bool]:
    manifest_value = response.get("remote_execution_manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ComputePolicyError("remote_execution_manifest_required_for_remote_chain")
    result = run_remote_chain(Path(manifest_value), timeout_s)
    result_path = output_dir / "remote_chain_result.json"
    write_remote_chain_result(result_path, result)
    print(f"remote_chain_result_path={result_path}")
    print(f"remote_chain_status={result.payload['chain_status']}")
    return (result_path, result.ok)


def _run_amorphous_structure_prep(
    response: JsonMap,
    output_dir: Path,
    timeout_s: float | None,
) -> tuple[Path, bool]:
    plan_value = response.get("amorphous_structure_prep_remote_plan_path")
    if not isinstance(plan_value, str) or not plan_value:
        raise ComputePolicyError("amorphous_structure_prep_remote_plan_required")
    result = run_remote_execution_plan(Path(plan_value), timeout_s)
    result_path = output_dir / "amorphous_structure_prep_remote_result.json"
    write_remote_execution_plan_result(result_path, result)
    print(f"amorphous_structure_prep_remote_result_path={result_path}")
    print(f"amorphous_structure_prep_remote_status={result.payload['plan_status']}")
    return (result_path, result.ok)
