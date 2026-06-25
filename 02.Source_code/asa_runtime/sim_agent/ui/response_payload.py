from __future__ import annotations

import json
from pathlib import Path

from sim_agent.runner import OfflineRunResult
from sim_agent.runner.artifact_contract import RUN_ARTIFACT_DESCRIPTORS
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str

from .api import UiApiStatus, UiApiValidation
from .model_auth import model_auth_status_payload
from .readiness import offline_production_readiness_payload


def status_payload(status: UiApiStatus) -> JsonMap:
    return {
        "static_root": str(status.static_root),
        "route_paths": list(status.route_paths),
        "offline_fixtures": list(status.offline_fixtures),
        "model_providers": list(status.model_providers),
        "model_options": list(status.model_options),
        "auth_modes": list(status.auth_modes),
        "agent_roles": list(status.agent_roles),
        "compute_targets": list(status.compute_targets),
        "model_auth": model_auth_status_payload(include_provider_credential_store=False),
        "graphdb": {
            "database_name": status.graphdb_database_name,
            "write_requires_approval": status.graphdb_write_requires_approval,
        },
    }


def validation_payload(validation: UiApiValidation) -> JsonMap:
    return {
        "can_run": validation.can_run,
        "missing_fields": list(validation.missing_fields),
        "compute_target": validation.compute_target,
        "runner_command": list(validation.runner_command),
    }


def run_response_payload(validation: UiApiValidation, result: OfflineRunResult) -> JsonMap:
    readiness = offline_production_readiness_payload(result)
    return {
        "can_run": validation.can_run,
        "missing_fields": list(validation.missing_fields),
        "compute_target": validation.compute_target,
        "runner_command": list(validation.runner_command),
        "run_status": result.run_status,
        "manifest_path": _display_path(result.manifest_path),
        "timeline_path": _display_path(result.timeline_path),
        "transport_field_path": _display_path(result.transport_field_path),
        "hit_history_path": _display_path(result.hit_history_path),
        "click_index_path": _display_path(result.click_index_path),
        "uncertainty_map_path": _display_path(result.uncertainty_map_path),
        "active_learning_plan_path": _display_path(result.active_learning_plan_path),
        "qa_report_path": _display_path(result.qa_report_path),
        "artifact_count": result.artifact_count,
        "agent_statuses": _agent_statuses(validation, result, readiness),
        "agent_message_log": _agent_message_log(result),
        "continuous_logs": _continuous_logs(result, readiness),
        "artifact_links": _artifact_links(result),
        "qa_report": _qa_report(result),
        "production_readiness": readiness,
        "bundle": _bundle_payload(result, readiness),
    }


def click_diagnostic_contract_payload() -> JsonMap:
    return {
        "fields": [
            "material_id",
            "region",
            "energy_transfer_eV",
            "damage_dose",
            "removed_depth_nm",
            "profile_history_nm",
            "energy_history_eV",
            "uncertainty_ood",
        ]
    }


def _bundle_payload(result: OfflineRunResult, readiness: JsonMap) -> JsonMap:
    return {
        "manifest": _read_json(result.manifest_path, "manifest"),
        "timeline": _read_json(result.timeline_path, "timeline"),
        "diagnostics": _read_json(result.click_index_path, "diagnostics"),
        "uncertainty_map": _read_json(result.uncertainty_map_path, "uncertainty_map"),
        "active_learning_plan": _read_json(result.active_learning_plan_path, "active_learning_plan"),
        "surrogate_training_gate": _read_json(result.output_dir / "surrogate_training_gate_report.json", "gate"),
        "surrogate_model_manifest": _read_json(result.output_dir / "surrogate_model_manifest.json", "manifest"),
        "qa_report": _qa_report(result),
        "production_readiness": readiness,
    }


def _read_json(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _qa_report(result: OfflineRunResult) -> JsonMap:
    return _read_json(result.qa_report_path, "qa_report")


def _agent_statuses(validation: UiApiValidation, result: OfflineRunResult, readiness: JsonMap) -> list[JsonMap]:
    qa = _qa_report(result)
    qa_status = as_str(qa.get("status", "blocked"), "qa_report.status")
    readiness_status = "blocked" if readiness.get("production_ready") is not True else "ready"
    return [
        _agent_status(
            "orchestrator",
            "Orchestrator",
            "complete",
            f"Validated {validation.request.mode} {validation.request.run_id} and dispatched offline run.",
            "Owns routing, approvals, and final run assembly.",
        ),
        _agent_status(
            "research_agent",
            "Research Agent",
            "ready",
            "GraphDB writes remain approval-gated; source-backed retrieval is available.",
            "Builds source-to-graph import artifacts and provenance-backed answers when invoked.",
        ),
        _agent_status(
            "md_agent",
            "MD Agent",
            "complete",
            "Loaded MD event fixture and preserved incident history for diagnostics.",
            "Production MD remains gated by force-field, box-size, event-count, and postprocess checks.",
        ),
        _agent_status(
            "ml_agent",
            "ML Agent",
            "complete",
            "Built empirical MDN artifact, accepted training gate, and active-learning report.",
            "Surrogate use is blocked in production unless the training gate is accepted.",
        ),
        _agent_status(
            "feature_scale_agent",
            "Feature Scale",
            result.run_status,
            "Ran transport and Level-Set profile timeline artifacts.",
            "Consumes MDN interaction outputs and plasma distributions to evolve the profile.",
        ),
        _agent_status(
            "qa_agent",
            "QA Agent",
            qa_status,
            f"QA report: {qa_status}; hard blockers={len(as_sequence(qa.get('hard_blockers', []), 'qa.hard_blockers'))}.",
            "Checks profile timeline, position-resolved energy, click diagnostics, and process time scale.",
        ),
        _agent_status(
            "production_gate",
            "Production Gate",
            readiness_status,
            f"Production readiness: {readiness_status}; blockers={len(_strings(readiness, 'hard_blockers'))}.",
            "Blocks go-live until real endpoint, approved compute, MDN, Level-Set, and Neo4j evidence exist.",
        ),
    ]


def _agent_status(agent_id: str, label: str, status: str, summary: str, detail: str) -> JsonMap:
    return {
        "agent_id": agent_id,
        "label": label,
        "status": status,
        "summary": summary,
        "detail": detail,
    }


def _agent_message_log(result: OfflineRunResult) -> list[JsonMap]:
    return [
        _message("orchestrator", "md_agent", "Use source-backed MD event data and preserve incident diagnostics."),
        _message("md_agent", "ml_agent", "MD events ready for interaction-kernel inference."),
        _message("ml_agent", "feature_scale_agent", "Kernel and uncertainty handoff ready."),
        _message("feature_scale_agent", "qa_agent", f"Profile timeline written: {result.timeline_path.name}."),
        _message("qa_agent", "orchestrator", f"QA report written: {result.qa_report_path.name}."),
        _message("orchestrator", "production_gate", "Evaluate go-live evidence after QA and external approvals."),
    ]


def _message(sender: str, recipient: str, message: str) -> JsonMap:
    return {"sender": sender, "recipient": recipient, "message": message}


def _continuous_logs(result: OfflineRunResult, readiness: JsonMap) -> list[str]:
    return [
        f"run_status={result.run_status}",
        f"manifest_path={_display_path(result.manifest_path)}",
        f"profile_timeline_path={_display_path(result.timeline_path)}",
        f"click_index_path={_display_path(result.click_index_path)}",
        f"surrogate_model_path={_display_path(result.output_dir / 'empirical_mdn_model.json')}",
        f"surrogate_training_gate_path={_display_path(result.output_dir / 'surrogate_training_gate_report.json')}",
        f"qa_report_path={_display_path(result.qa_report_path)}",
        f"production_ready={str(readiness.get('production_ready') is True).lower()}",
        "model_settings_scope=agent_plan_only_for_offline_fixture",
    ]


def _artifact_links(result: OfflineRunResult) -> JsonMap:
    path_by_key = {
        "manifest": result.manifest_path,
        "profile_timeline": result.timeline_path,
        "transport_field": result.transport_field_path,
        "hit_history": result.hit_history_path,
        "click_index": result.click_index_path,
        "uncertainty_map": result.uncertainty_map_path,
        "active_learning_plan": result.active_learning_plan_path,
        "qa_report": result.qa_report_path,
    }
    return {
        descriptor.key: _display_path(path_by_key.get(descriptor.key, result.output_dir / descriptor.filename))
        for descriptor in RUN_ARTIFACT_DESCRIPTORS
    }


def _display_path(path: Path) -> str:
    source_root = Path(__file__).resolve().parents[2].resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(source_root).as_posix()
    except ValueError:
        return resolved.name


def _strings(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]
