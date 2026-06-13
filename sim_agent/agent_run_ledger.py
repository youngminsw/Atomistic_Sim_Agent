from __future__ import annotations

import json
from pathlib import Path

from sim_agent.md import assess_md_production_readiness
from sim_agent.agent_run_quality import build_agent_run_quality
from sim_agent.schemas._parse import JsonMap


AGENT_RUN_LEDGER_NAME = "agent_run_ledger.json"


def write_agent_run_ledger(
    output_dir: Path,
    request_payload: JsonMap,
    compute_response: JsonMap,
    amorphous_prep_result_path: Path | None,
    capability_result_path: Path | None,
    chain_result_path: Path | None,
    surrogate_gate_result_path: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    amorphous_prep_result = _read_optional_json(amorphous_prep_result_path)
    capability_result = _read_optional_json(capability_result_path)
    chain_result = _read_optional_json(chain_result_path)
    surrogate_gate_result = _read_optional_json(surrogate_gate_result_path)
    md_readiness = assess_md_production_readiness(request_payload, compute_response)
    quality = build_agent_run_quality(
        compute_response,
        amorphous_prep_result,
        capability_result,
        chain_result,
        surrogate_gate_result,
        md_readiness.payload,
    )
    ledger_path = output_dir / AGENT_RUN_LEDGER_NAME
    payload = _ledger_payload(
        output_dir,
        request_payload,
        compute_response,
        md_readiness.payload,
        quality.overall_status,
        quality.qa_payload,
        quality.evidence,
        amorphous_prep_result_path,
        amorphous_prep_result,
        capability_result_path,
        capability_result,
        chain_result_path,
        chain_result,
        surrogate_gate_result_path,
        surrogate_gate_result,
    )
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger_path


def _ledger_payload(
    output_dir: Path,
    request_payload: JsonMap,
    compute_response: JsonMap,
    md_readiness: JsonMap,
    overall_status: str,
    qa_payload: JsonMap,
    evidence: list[str],
    amorphous_prep_result_path: Path | None,
    amorphous_prep_result: JsonMap,
    capability_result_path: Path | None,
    capability_result: JsonMap,
    chain_result_path: Path | None,
    chain_result: JsonMap,
    surrogate_gate_result_path: Path | None,
    surrogate_gate_result: JsonMap,
) -> JsonMap:
    return {
        "ledger_version": "agent_run_ledger_v1",
        "request_id": _text(request_payload, "request_id"),
        "run_id": _text(compute_response, "run_id"),
        "artifact_dir": _text_or_default(compute_response, "artifact_dir", str(output_dir)),
        "overall_status": overall_status,
        "pipeline_stages": _pipeline_stages(surrogate_gate_result),
        "compute_target": _compute_target(compute_response),
        "artifact_paths": _artifact_paths(
            compute_response,
            amorphous_prep_result_path,
            capability_result_path,
            chain_result_path,
            surrogate_gate_result_path,
        ),
        "model_provider": _model_provider(request_payload),
        "md": md_readiness,
        "remote": {
            "amorphous_prep_status": _text(amorphous_prep_result, "plan_status"),
            "amorphous_prep_blockers": _string_list(amorphous_prep_result, "blockers"),
            "capability_probe_status": _text(capability_result, "probe_status"),
            "chain_status": _text(chain_result, "chain_status"),
            "capability_blockers": _string_list(capability_result, "blockers"),
            "chain_blockers": _string_list(chain_result, "blockers"),
            "completed_stage_ids": _string_list(chain_result, "completed_stage_ids"),
            "missing_stage_ids": _string_list(chain_result, "missing_stage_ids"),
        },
        "graphdb": _graphdb_summary(compute_response),
        "surrogate": {
            "training_gate_present": bool(surrogate_gate_result),
            "training_gate_decision": _text(surrogate_gate_result, "decision"),
            "training_gate_accepted": _optional_bool(surrogate_gate_result, "accepted"),
            "training_gate_blockers": _string_list(surrogate_gate_result, "blockers"),
            "training_gate_evidence": _string_list(surrogate_gate_result, "evidence"),
            "next_actions": _string_list(surrogate_gate_result, "next_actions"),
        },
        "qa": qa_payload,
        "evidence": evidence,
    }


def _compute_target(compute_response: JsonMap) -> JsonMap:
    worker = compute_response.get("worker_bundle")
    payload: dict[str, object] = {}
    if isinstance(worker, dict):
        for source_field, target_field in (
            ("host_alias", "host"),
            ("environment_name", "environment_name"),
        ):
            value = worker.get(source_field)
            if isinstance(value, str) and value:
                payload[target_field] = value
    manifest = compute_response.get("remote_execution_manifest")
    if isinstance(manifest, dict):
        ssh_target = manifest.get("ssh_target")
        ssh_port = manifest.get("ssh_port")
        if isinstance(ssh_target, str) and ssh_target:
            payload["ssh_target"] = ssh_target
        if isinstance(ssh_port, int) and not isinstance(ssh_port, bool):
            payload["ssh_port"] = ssh_port
    return payload


def _artifact_paths(
    compute_response: JsonMap,
    amorphous_prep_result_path: Path | None,
    capability_result_path: Path | None,
    chain_result_path: Path | None,
    surrogate_gate_result_path: Path | None,
) -> JsonMap:
    keys = (
        "manifest_path",
        "md_campaign_plan_path",
        "validated_request_path",
        "source_payload_path",
        "amorphous_structure_prep_manifest_path",
        "amorphous_structure_source_path",
        "amorphous_structure_prep_job_path",
        "amorphous_structure_prep_worker_path",
        "amorphous_structure_prep_remote_plan_path",
        "worker_path",
        "lammps_execution_worker_path",
        "md_postprocess_worker_path",
        "remote_execution_chain_path",
        "remote_execution_script_path",
        "remote_execution_manifest_path",
        "graphdb_agent_report_path",
        "graphdb_import_bundle_dir",
        "graphdb_ingest_report_path",
        "graphdb_retrieval_context_path",
        "graphdb_agent_context_path",
        "research_answer_path",
    )
    paths = {key: _text(compute_response, key) for key in keys if _text(compute_response, key)}
    if amorphous_prep_result_path is not None:
        paths["amorphous_structure_prep_remote_result_path"] = str(
            amorphous_prep_result_path
        )
    if capability_result_path is not None:
        paths["remote_capability_probe_result_path"] = str(capability_result_path)
    if chain_result_path is not None:
        paths["remote_chain_result_path"] = str(chain_result_path)
    if surrogate_gate_result_path is not None:
        paths["surrogate_training_gate_result_path"] = str(surrogate_gate_result_path)
    return paths


def _graphdb_summary(compute_response: JsonMap) -> JsonMap:
    report = compute_response.get("graphdb_agent_report")
    if not isinstance(report, dict):
        return {"status": "", "ingest_accepted": None}
    bundle = report.get("bundle")
    bundle_report = bundle.get("report") if isinstance(bundle, dict) else None
    if not isinstance(bundle_report, dict):
        bundle_report = {}
    return {
        "status": _text(report, "status"),
        "write": _optional_bool(report, "graphdb_write"),
        "ingest_accepted": _optional_bool(bundle_report, "accepted"),
        "database_name": _text(bundle_report, "database_name"),
        "source_count": _optional_int(bundle_report, "source_count"),
        "claim_count": _optional_int(bundle_report, "claim_count"),
        "entity_count": _optional_int(bundle_report, "entity_count"),
        "blocker_reasons": _string_list(bundle_report, "blocker_reasons"),
    }


def _pipeline_stages(surrogate_gate_result: JsonMap) -> list[str]:
    stages = [
        "agent_plan",
        "md_campaign_worker_bundle",
        "lammps_execution_worker_bundle",
        "md_postprocess_worker_bundle",
    ]
    if surrogate_gate_result:
        stages.append("surrogate_training_gate")
    return stages


def _model_provider(request_payload: JsonMap) -> JsonMap:
    endpoint = request_payload.get("llm_endpoint")
    if not isinstance(endpoint, dict):
        return {}
    fields = (
        "provider",
        "model",
        "reasoning_effort",
        "base_url",
        "use_case",
        "structured_outputs",
        "streaming",
        "api_key_env",
        "auth_mode",
    )
    return {field: endpoint[field] for field in fields if field in endpoint}


def _read_optional_json(path: Path | None) -> JsonMap:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "blockers": ["result_json_unreadable"]}
    if isinstance(payload, dict):
        return payload
    return {"ok": False, "blockers": ["result_json_not_object"]}


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def _text_or_default(payload: JsonMap, field: str, default: str) -> str:
    value = _text(payload, field)
    return value if value else default


def _string_list(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _optional_bool(payload: JsonMap, field: str) -> bool | None:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    return None


def _optional_int(payload: JsonMap, field: str) -> int | None:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
