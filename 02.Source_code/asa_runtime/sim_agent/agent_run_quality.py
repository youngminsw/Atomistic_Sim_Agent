from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class AgentRunQuality:
    overall_status: str
    qa_payload: JsonMap
    evidence: list[str]


def build_agent_run_quality(
    compute_response: JsonMap,
    amorphous_prep_result: JsonMap,
    capability_result: JsonMap,
    chain_result: JsonMap,
    surrogate_gate_result: JsonMap,
    md_readiness: JsonMap,
) -> AgentRunQuality:
    hard_blockers = _qa_hard_blockers(
        amorphous_prep_result,
        capability_result,
        chain_result,
        surrogate_gate_result,
        md_readiness,
    )
    return AgentRunQuality(
        overall_status=_overall_status(
            amorphous_prep_result,
            capability_result,
            chain_result,
            surrogate_gate_result,
            md_readiness,
        ),
        qa_payload=_qa_payload(hard_blockers),
        evidence=_evidence(
            compute_response,
            amorphous_prep_result,
            capability_result,
            chain_result,
            surrogate_gate_result,
            md_readiness,
        ),
    )


def _qa_payload(hard_blockers: list[str]) -> JsonMap:
    return {
        "agent_id": "qa_agent",
        "evidence_scope": "planning_ledger_gate",
        "status": "pass" if not hard_blockers else "blocked",
        "hard_blockers": hard_blockers,
        "required_evidence": [
            "md_500_incident_execution_or_planned_worker_bundle",
            "md_physics_gate",
            "surrogate_training_gate",
            "level_set_profile_timeline",
            "graphdb_ingest_report",
        ],
        "report_summary": "hard blockers clear" if not hard_blockers else "hard blockers require resolution",
    }


def _qa_hard_blockers(
    amorphous_prep_result: JsonMap,
    capability_result: JsonMap,
    chain_result: JsonMap,
    surrogate_gate_result: JsonMap,
    md_readiness: JsonMap,
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(_string_list(md_readiness, "hard_blockers"))
    if _is_false(amorphous_prep_result, "ok"):
        blockers.extend(
            _string_list(amorphous_prep_result, "blockers")
            or ["amorphous_structure_prep_remote_failed"]
        )
    if _is_false(capability_result, "ok"):
        blockers.extend(_string_list(capability_result, "blockers") or ["remote_capability_probe_failed"])
    if _is_false(chain_result, "ok"):
        blockers.extend(_string_list(chain_result, "blockers") or ["remote_chain_failed"])
    if surrogate_gate_result and not _is_true(surrogate_gate_result, "accepted"):
        blockers.extend(_string_list(surrogate_gate_result, "blockers") or ["surrogate_training_gate_not_accepted"])
    return blockers


def _overall_status(
    amorphous_prep_result: JsonMap,
    capability_result: JsonMap,
    chain_result: JsonMap,
    surrogate_gate_result: JsonMap,
    md_readiness: JsonMap,
) -> str:
    if (
        _is_false(amorphous_prep_result, "ok")
        or _is_false(chain_result, "ok")
        or _is_false(capability_result, "ok")
    ):
        return "remote_failed"
    if surrogate_gate_result and not _is_true(surrogate_gate_result, "accepted"):
        return "surrogate_action_required"
    if _string_list(md_readiness, "hard_blockers"):
        return "md_action_required"
    if _is_true(chain_result, "ok"):
        return "remote_chain_completed"
    if _is_true(capability_result, "ok"):
        return "remote_capability_ready"
    return "planned"


def _evidence(
    compute_response: JsonMap,
    amorphous_prep_result: JsonMap,
    capability_result: JsonMap,
    chain_result: JsonMap,
    surrogate_gate_result: JsonMap,
    md_readiness: JsonMap,
) -> list[str]:
    evidence = ["agent_plan_artifacts_written"]
    if md_readiness:
        evidence.append("md_production_readiness_recorded")
    if _is_true(md_readiness, "production_ready"):
        evidence.append("md_production_ready")
    if _text(compute_response, "lammps_execution_worker_path"):
        evidence.append("lammps_execution_worker_bundle_written")
    if _text(compute_response, "amorphous_structure_prep_worker_path"):
        evidence.append("amorphous_structure_prep_worker_bundle_written")
    if amorphous_prep_result:
        evidence.append("amorphous_structure_prep_remote_result_recorded")
    if _is_true(amorphous_prep_result, "ok"):
        evidence.append("amorphous_structure_prep_remote_completed")
    if _text(compute_response, "remote_execution_manifest_path"):
        evidence.append("remote_execution_manifest_written")
    if _text(compute_response, "graphdb_import_bundle_dir"):
        evidence.append("graphdb_import_bundle_written")
    if capability_result:
        evidence.append("remote_capability_probe_result_recorded")
    if chain_result:
        evidence.append("remote_chain_result_recorded")
    if surrogate_gate_result:
        evidence.append("surrogate_training_gate_result_recorded")
    if _is_true(surrogate_gate_result, "accepted"):
        evidence.append("surrogate_training_gate_accepted")
    return evidence


def _text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return ""


def _string_list(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _is_true(payload: JsonMap, field: str) -> bool:
    return payload.get(field) is True


def _is_false(payload: JsonMap, field: str) -> bool:
    return payload.get(field) is False
