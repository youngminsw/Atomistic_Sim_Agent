from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sim_agent.schemas._parse import JsonMap

from .domain_adapters import (
    feature_scale_skill_adapter,
    md_skill_adapter,
    ml_skill_adapter,
    orchestrator_skill_adapter,
    qa_skill_adapter,
    research_skill_adapter,
)
from .invocation_artifacts import write_skill_invocation_artifact
from .types import SkillInvocationResult


@dataclass(frozen=True, slots=True)
class AgentSkillContract:
    agent_id: str
    skill_id: str
    domain_adapter: str
    purpose: str
    required_inputs: tuple[str, ...]
    expected_artifacts: tuple[str, ...]


SkillHandler = Callable[[JsonMap, AgentSkillContract], SkillInvocationResult]


def agent_skill_contracts() -> tuple[AgentSkillContract, ...]:
    return (
        AgentSkillContract(
            "orchestrator",
            "orchestrate_simulation_run",
            "agent_harness.orchestrator",
            "Route the user goal through research, MD, ML/MDN, feature-scale, and QA agents.",
            ("user_goal", "request_id"),
            ("agent_run_ledger.json", "agent_team_session_ledger.json"),
        ),
        AgentSkillContract(
            "md_agent",
            "prepare_and_verify_lammps_md",
            "md.execution_runner",
            "Prepare LAMMPS MD campaigns and enforce physical event-quality gates.",
            ("material", "phase", "ion", "md_incident_count"),
            ("md_campaign_manifest.json", "md_physics_gate_report.json"),
        ),
        AgentSkillContract(
            "ml_agent",
            "train_and_gate_mdn_surrogate",
            "ml_surrogate.training_gate",
            "Train or reject MD-derived MDN surrogates through quantitative gates.",
            ("md_events_path", "surrogate_training_gate"),
            ("surrogate_model_manifest.json", "surrogate_training_gate_report.json"),
        ),
        AgentSkillContract(
            "feature_scale_agent",
            "run_feature_scale_level_set",
            "level_set.transport_evolution",
            "Use MDN interaction outputs in transport and Level-Set profile evolution.",
            ("geometry_path", "iedf", "iadf", "process_time_s"),
            ("profile_timeline.json", "click_diagnostics.json"),
        ),
        AgentSkillContract(
            "research_agent",
            "research_and_ingest_graphdb_catalog",
            "knowledge.research_agent",
            "Map source literature and code samples into the project Neo4j catalog with provenance.",
            ("research_question", "graphdb_mode"),
            ("source_catalog.json", "graph_import_plan.json"),
        ),
        AgentSkillContract(
            "qa_agent",
            "qa_physics_and_runtime_evidence",
            "agent_run_quality",
            "Reject runs that miss physics, MDN, Level-Set, controller, or GraphDB gates.",
            ("agent_run_ledger", "quality_gates"),
            ("qa_report.json", "production_readiness_report.json"),
        ),
    )


def skill_registry_summary() -> JsonMap:
    return {
        "registry_version": "asa_agent_skill_registry_v1",
        "dispatch_mode": "callable_handlers",
        "skills": [_contract_payload(contract) for contract in agent_skill_contracts()],
    }


def run_registered_agent_skills(payload: JsonMap, output_dir: Path | None = None) -> tuple[SkillInvocationResult, ...]:
    handlers = _handlers()
    invocations: list[SkillInvocationResult] = []
    for contract in agent_skill_contracts():
        handler = handlers[contract.skill_id]
        invocation = handler(payload, contract)
        if output_dir is not None:
            write_skill_invocation_artifact(output_dir, invocation)
        invocations.append(invocation)
    return tuple(invocations)


def run_registered_agent_skill(skill_id: str, payload: JsonMap, output_dir: Path | None = None) -> SkillInvocationResult | None:
    handlers = _handlers()
    handler = handlers.get(skill_id)
    if handler is None:
        return None
    contract = next((item for item in agent_skill_contracts() if item.skill_id == skill_id), None)
    if contract is None:
        return None
    invocation = handler(payload, contract)
    if output_dir is not None:
        write_skill_invocation_artifact(output_dir, invocation)
    return invocation


def _handlers() -> dict[str, SkillHandler]:
    return {
        "orchestrate_simulation_run": _orchestrator_handler,
        "prepare_and_verify_lammps_md": _md_handler,
        "train_and_gate_mdn_surrogate": _ml_handler,
        "run_feature_scale_level_set": _feature_scale_handler,
        "research_and_ingest_graphdb_catalog": _research_handler,
        "qa_physics_and_runtime_evidence": _qa_handler,
    }


def _orchestrator_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=orchestrator_skill_adapter(payload),
        next_action="route_specialist_calls_after_preflight",
    )


def _md_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=md_skill_adapter(payload),
        next_action="request_qa_before_lammps_or_slurm_submission",
    )


def _ml_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=ml_skill_adapter(payload),
        next_action="train_mdn_or_request_active_learning_md",
    )


def _feature_scale_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=feature_scale_skill_adapter(payload),
        next_action="run_profile_timeline_after_surrogate_acceptance",
    )


def _research_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=research_skill_adapter(payload),
        next_action="ingest_after_user_approved_empty_database",
    )


def _qa_handler(payload: JsonMap, contract: AgentSkillContract) -> SkillInvocationResult:
    return _adapter_result(
        payload,
        contract,
        adapter_output=qa_skill_adapter(payload),
        next_action="approve_or_return_blockers_to_owner_agent",
    )


def _adapter_result(
    payload: JsonMap,
    contract: AgentSkillContract,
    *,
    adapter_output: JsonMap,
    next_action: str,
) -> SkillInvocationResult:
    request_id = payload.get("request_id")
    request_slug = request_id if isinstance(request_id, str) and request_id else "anonymous"
    missing = _missing_inputs(payload, contract.required_inputs)
    adapter_blockers = _adapter_blockers(adapter_output)
    ready = not missing and not adapter_blockers
    return SkillInvocationResult(
        agent_id=contract.agent_id,
        skill_id=contract.skill_id,
        status="ready" if ready else "blocked",
        execution_status="adapter_contract_ready" if ready else "adapter_preflight_blocked",
        domain_adapter=contract.domain_adapter,
        artifact_ref=f"skill_invocations/{request_slug}/{contract.agent_id}/{contract.skill_id}.json",
        contract=_contract_payload(contract),
        result={
            "adapter_invoked": True,
            "adapter_action": _adapter_action(adapter_output, contract.domain_adapter),
            "adapter_output": adapter_output,
            "required_inputs_present": ready,
            "missing_inputs": list(missing),
            "adapter_blockers": list(adapter_blockers),
            "next_action": next_action,
        },
    )


def _missing_inputs(payload: JsonMap, required_inputs: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for field in required_inputs:
        value = payload.get(field)
        if value is None or value == "":
            missing.append(field)
    return tuple(missing)


def _adapter_action(adapter_output: JsonMap, fallback: str) -> str:
    value = adapter_output.get("adapter_action")
    if isinstance(value, str) and value:
        return value
    return fallback


def _adapter_blockers(adapter_output: JsonMap) -> tuple[str, ...]:
    value = adapter_output.get("adapter_blockers")
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _contract_payload(contract: AgentSkillContract) -> JsonMap:
    return {
        "agent_id": contract.agent_id,
        "skill_id": contract.skill_id,
        "domain_adapter": contract.domain_adapter,
        "purpose": contract.purpose,
        "required_inputs": list(contract.required_inputs),
        "expected_artifacts": list(contract.expected_artifacts),
    }
