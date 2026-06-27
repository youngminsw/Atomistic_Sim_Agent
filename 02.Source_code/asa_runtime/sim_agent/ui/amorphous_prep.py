from __future__ import annotations

import json
from pathlib import Path

from sim_agent.compute import (
    build_amorphous_structure_prep_job,
    build_remote_execution_plan,
    build_worker_bundle,
    job_bundle_payload,
    remote_execution_plan_manifest_payload,
    WorkerBundle,
    worker_bundle_payload,
)
from sim_agent.md import (
    AmorphousStructurePrepConfig,
    AmorphousStructurePrepError,
    stage_amorphous_structure_prep_bundle,
)
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


def maybe_stage_amorphous_structure_prep(
    request: JsonMap,
    output_dir: Path,
    repo_root: Path,
    run_id: str,
    host_alias: str,
    environment_name: str,
    remote_user: str,
    ssh_target: str | None,
    ssh_port: int | None,
) -> JsonMap:
    surface = _surface_state(request)
    if surface.get("phase") != "amorphous" or "lammps_structure_source" in surface:
        return {}
    material_id = as_str(require(surface, "material_id"), "surface_state.material_id")
    atom_count = _atom_count(surface)
    try:
        bundle = stage_amorphous_structure_prep_bundle(
            AmorphousStructurePrepConfig(
                material_id=material_id,
                atom_count=atom_count,
            ),
            output_dir / "amorphous_structure_prep",
            repo_root,
        )
    except AmorphousStructurePrepError as exc:
        raise SchemaValidationError(str(exc)) from exc
    job = build_amorphous_structure_prep_job(
        run_id=run_id,
        environment_name=environment_name,
        material_id=material_id,
        atom_count=atom_count,
    )
    worker = build_worker_bundle(host_alias, job, remote_user=remote_user)
    job_path = output_dir / "amorphous_structure_prep_job.json"
    worker_path = output_dir / "amorphous_structure_prep_worker_bundle.json"
    job_payload = job_bundle_payload(job)
    worker_payload = worker_bundle_payload(worker)
    _write_json(job_path, job_payload)
    _write_json(worker_path, worker_payload)
    return {
        "amorphous_structure_prep_manifest_path": str(bundle.manifest_path),
        "amorphous_structure_source_path": str(bundle.structure_source_path),
        "amorphous_structure_prep": bundle.manifest_payload,
        "amorphous_structure_source": bundle.structure_source_payload,
        "amorphous_structure_prep_job_path": str(job_path),
        "amorphous_structure_prep_worker_path": str(worker_path),
        "amorphous_structure_prep_job": job_payload,
        "amorphous_structure_prep_worker_bundle": worker_payload,
    } | _remote_plan_outputs(output_dir, worker, ssh_target, ssh_port)


def _surface_state(request: JsonMap) -> JsonMap:
    scene = as_mapping(require(request, "scene"), "scene")
    return as_mapping(require(scene, "surface_state"), "surface_state")


def _atom_count(surface: JsonMap) -> int:
    md_box = as_mapping(require(surface, "md_box"), "surface_state.md_box")
    value = md_box.get("atom_count")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise SchemaValidationError("surface_state.md_box.atom_count must be a positive integer")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remote_plan_outputs(
    output_dir: Path,
    worker: WorkerBundle,
    ssh_target: str | None,
    ssh_port: int | None,
) -> JsonMap:
    if ssh_target is None or ssh_port is None:
        return {}
    plan = build_remote_execution_plan(worker, ssh_target=ssh_target, ssh_port=ssh_port)
    remote_dir = output_dir / "remote"
    plan_path = remote_dir / "amorphous_structure_prep_remote_plan.json"
    remote_dir.mkdir(parents=True, exist_ok=True)
    payload = remote_execution_plan_manifest_payload(
        plan,
        source_root=Path(__file__).resolve().parents[2],
        output_root=output_dir,
    )
    _write_json(plan_path, payload)
    return {
        "amorphous_structure_prep_remote_plan_path": str(plan_path),
        "amorphous_structure_prep_remote_execution_plan": payload,
    }
