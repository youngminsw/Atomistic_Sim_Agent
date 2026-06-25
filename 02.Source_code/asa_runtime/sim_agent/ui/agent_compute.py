from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness import (
    AgentPlanArtifactError,
    OfflineModelClient,
    SimulationAgentHarness,
    write_agent_plan_artifacts,
)
from sim_agent.compute import (
    ComputePolicyError,
    WorkerBundle,
    build_lammps_execution_job_from_md_campaign_job,
    build_remote_execution_chain,
    build_md_campaign_job_from_plan_dir,
    build_md_postprocess_job_from_lammps_execution_job,
    build_remote_execution_plan,
    build_worker_bundle,
    job_bundle_payload,
    remote_execution_chain_payload,
    remote_execution_plan_payload,
    resolve_worker_host,
    stage_compute_source_payload,
    worker_bundle_payload,
    write_remote_execution_script_bundle,
)
from sim_agent.llm_endpoints import (
    ModelPolicyError,
    ModelProviderConfig,
    ProviderConfigPolicyError,
)
from sim_agent.model_provider_payload import model_provider_payload
from sim_agent.knowledge import (
    GraphDBGateRequest,
    GraphDBMode,
    ResearchQuestion,
    build_graphdb_gate_plan,
    build_research_agent_artifacts,
    research_agent_payload,
    seeded_provenance_registry,
)
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError

from .amorphous_prep import maybe_stage_amorphous_structure_prep


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def build_agent_compute_bundle_http_response(payload: JsonMap) -> tuple[JsonMap, int]:
    try:
        request = as_mapping(require(payload, "request"), "request")
        output_dir = Path(as_str(require(payload, "output_dir"), "output_dir"))
        host_alias = as_str(require(payload, "host"), "host")
        environment_name = as_str(require(payload, "environment_name"), "environment_name")
        remote_user = _optional_str(payload, "remote_user")
        ssh_target = _optional_str(payload, "ssh_target")
        ssh_port = _optional_int(payload, "ssh_port")
        md_incident_count = _optional_int(payload, "md_incident_count")
        if md_incident_count is None:
            md_incident_count = 500
        host = resolve_worker_host(host_alias, environment_name, remote_user, ssh_target, ssh_port)
        return (
            _prepare_bundle(
                request,
                output_dir,
                host.host_alias,
                host.environment_name,
                host.remote_user,
                host.ssh_target,
                host.ssh_port,
                md_incident_count,
            ),
            200,
        )
    except (
        AgentPlanArtifactError,
        ComputePolicyError,
        ModelPolicyError,
        ProviderConfigPolicyError,
        OSError,
        SchemaValidationError,
    ) as exc:
        return {"error": str(exc)}, 400


def _prepare_bundle(
    request: JsonMap,
    output_dir: Path,
    host_alias: str,
    environment_name: str,
    remote_user: str,
    ssh_target: str | None,
    ssh_port: int | None,
    md_incident_count: int,
) -> JsonMap:
    endpoint = ModelProviderConfig.from_mapping(model_provider_payload(request))
    result = SimulationAgentHarness(
        endpoint=endpoint,
        client=OfflineModelClient(),
    ).plan(request)
    artifact_bundle = write_agent_plan_artifacts(output_dir, request, result)
    amorphous_prep = maybe_stage_amorphous_structure_prep(
        request,
        output_dir,
        SOURCE_ROOT,
        result.run_id,
        host_alias,
        environment_name,
        remote_user,
        ssh_target,
        ssh_port,
    )
    source_payload = stage_compute_source_payload(SOURCE_ROOT, output_dir)
    job = build_md_campaign_job_from_plan_dir(
        output_dir,
        environment_name,
        incident_count=md_incident_count,
    )
    worker = build_worker_bundle(host_alias, job, remote_user=remote_user)
    lammps_job = build_lammps_execution_job_from_md_campaign_job(job)
    lammps_worker = build_worker_bundle(host_alias, lammps_job, remote_user=remote_user)
    postprocess_job = build_md_postprocess_job_from_lammps_execution_job(
        lammps_job,
        material_id="Si",
    )
    postprocess_worker = build_worker_bundle(
        host_alias,
        postprocess_job,
        remote_user=remote_user,
    )
    job_path = output_dir / "md_campaign_job.json"
    worker_path = output_dir / "worker_bundle.json"
    lammps_job_path = output_dir / "lammps_execution_job.json"
    lammps_worker_path = output_dir / "lammps_execution_worker_bundle.json"
    postprocess_job_path = output_dir / "md_postprocess_job.json"
    postprocess_worker_path = output_dir / "md_postprocess_worker_bundle.json"
    job_payload = job_bundle_payload(job)
    worker_payload = worker_bundle_payload(worker)
    lammps_job_payload = job_bundle_payload(lammps_job)
    lammps_worker_payload = worker_bundle_payload(lammps_worker)
    postprocess_job_payload = job_bundle_payload(postprocess_job)
    postprocess_worker_payload = worker_bundle_payload(postprocess_worker)
    _write_json(job_path, job_payload)
    _write_json(worker_path, worker_payload)
    _write_json(lammps_job_path, lammps_job_payload)
    _write_json(lammps_worker_path, lammps_worker_payload)
    _write_json(postprocess_job_path, postprocess_job_payload)
    _write_json(postprocess_worker_path, postprocess_worker_payload)
    payload: dict[str, object] = {
        "prepared": True,
        "run_id": result.run_id,
        "artifact_dir": str(output_dir),
        "manifest_path": str(artifact_bundle.manifest_path),
        "md_campaign_plan_path": str(artifact_bundle.md_campaign_plan_path),
        "validated_request_path": str(artifact_bundle.validated_request_path),
        "source_payload_path": str(source_payload.archive_path),
        "source_payload_manifest": source_payload.manifest_payload,
        "job_path": str(job_path),
        "worker_path": str(worker_path),
        "job": job_payload,
        "worker_bundle": worker_payload,
        "lammps_execution_job_path": str(lammps_job_path),
        "lammps_execution_worker_path": str(lammps_worker_path),
        "lammps_execution_job": lammps_job_payload,
        "lammps_execution_worker_bundle": lammps_worker_payload,
        "md_postprocess_job_path": str(postprocess_job_path),
        "md_postprocess_worker_path": str(postprocess_worker_path),
        "md_postprocess_job": postprocess_job_payload,
        "md_postprocess_worker_bundle": postprocess_worker_payload,
    }
    payload.update(amorphous_prep)
    payload.update(_research_outputs(output_dir, result.run_id))
    payload.update(_remote_plan_outputs(output_dir, worker, ssh_target, ssh_port))
    payload.update(_lammps_remote_plan_outputs(output_dir, lammps_worker, ssh_target, ssh_port))
    payload.update(
        _remote_chain_outputs(
            output_dir,
            (worker, lammps_worker, postprocess_worker),
            ssh_target,
            ssh_port,
        )
    )
    return payload


def _research_outputs(output_dir: Path, run_id: str) -> JsonMap:
    graph_dir = output_dir / "research_graph"
    gate_plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=(),
            database_name="atomistic_sim_agent_knowledge",
            requires_empty_database=True,
        )
    )
    result = build_research_agent_artifacts(
        seeded_provenance_registry(),
        gate_plan,
        graph_dir,
        sync_run_id=f"{run_id}-research-graph",
        question=ResearchQuestion(
            query="What source-backed knowledge should agents read before simulation?",
            tags=(),
        ),
    )
    report_path = output_dir / "research_agent.json"
    payload = research_agent_payload(result)
    _write_json(report_path, payload)
    return {
        "graphdb_agent_report_path": str(report_path),
        "graphdb_import_bundle_dir": str(result.bundle.output_dir),
        "graphdb_ingest_report_path": str(result.bundle.ingest_report_path),
        "graphdb_retrieval_context_path": str(result.bundle.retrieval_context_path),
        "graphdb_agent_context_path": str(result.agent_context_path),
        "research_answer_path": str(result.answer_path),
        "graphdb_agent_report": payload,
    }


def _remote_plan_outputs(
    output_dir: Path,
    worker: WorkerBundle,
    ssh_target: str | None,
    ssh_port: int | None,
) -> JsonMap:
    if ssh_target is None or ssh_port is None:
        return {}
    plan = build_remote_execution_plan(worker, ssh_target=ssh_target, ssh_port=ssh_port)
    plan_path = output_dir / "remote_plan.json"
    payload = remote_execution_plan_payload(plan)
    _write_json(plan_path, payload)
    return {
        "remote_plan_path": str(plan_path),
        "remote_execution_plan": payload,
    }


def _lammps_remote_plan_outputs(
    output_dir: Path,
    worker: WorkerBundle,
    ssh_target: str | None,
    ssh_port: int | None,
) -> JsonMap:
    if ssh_target is None or ssh_port is None:
        return {}
    plan = build_remote_execution_plan(worker, ssh_target=ssh_target, ssh_port=ssh_port)
    plan_path = output_dir / "lammps_remote_plan.json"
    payload = remote_execution_plan_payload(plan)
    _write_json(plan_path, payload)
    return {
        "lammps_remote_plan_path": str(plan_path),
        "lammps_remote_execution_plan": payload,
    }


def _remote_chain_outputs(
    output_dir: Path,
    workers: tuple[WorkerBundle, ...],
    ssh_target: str | None,
    ssh_port: int | None,
) -> JsonMap:
    if ssh_target is None or ssh_port is None:
        return {}
    chain = build_remote_execution_chain(workers, ssh_target=ssh_target, ssh_port=ssh_port)
    chain_path = output_dir / "remote_chain.json"
    script_path = output_dir / "remote_chain.sh"
    manifest_path = output_dir / "remote_chain_manifest.json"
    payload = remote_execution_chain_payload(chain)
    script_bundle = write_remote_execution_script_bundle(
        chain,
        script_path=script_path,
        manifest_path=manifest_path,
    )
    _write_json(chain_path, payload)
    return {
        "remote_execution_chain_path": str(chain_path),
        "remote_execution_chain": payload,
        "remote_execution_script_path": str(script_bundle.script_path),
        "remote_execution_manifest_path": str(script_bundle.manifest_path),
        "remote_execution_manifest": script_bundle.manifest_payload,
    }


def _optional_str(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    return as_str(value, field)


def _optional_int(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be an integer")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
