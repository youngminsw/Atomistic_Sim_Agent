from __future__ import annotations

from collections.abc import Sequence

from sim_agent.md_campaign.planner import plan_md_campaign
from sim_agent.md_campaign.serialize import md_campaign_plan_payload
from sim_agent.md_campaign.types import CampaignPlanError
from sim_agent.schemas._parse import JsonMap


def orchestrator_skill_adapter(payload: JsonMap) -> JsonMap:
    return {
        "adapter_action": "agent_harness.prepare_run_plan",
        "artifacts": ("agent_run_ledger.json", "agent_team_session_ledger.json"),
        "specialist_call_order": (
            "research_agent",
            "md_agent",
            "ml_agent",
            "feature_scale_agent",
            "qa_agent",
        ),
        "run_goal": _text(payload, "user_goal", "simulation_run"),
    }


def md_skill_adapter(payload: JsonMap) -> JsonMap:
    try:
        plan = plan_md_campaign(
            material_id=_text(payload, "material", "Si"),
            ion_species=_text(payload, "ion", "Ar"),
            phases=(_text(payload, "phase", "amorphous"),),
            energy_range_eV=_range(payload, "energy_range_eV", (30.0, 150.0)),
            polar_range_deg=_range(payload, "polar_range_deg", (0.0, 55.0)),
            azimuth_range_deg=_range(payload, "azimuth_range_deg", (0.0, 360.0)),
        )
    except CampaignPlanError as exc:
        return {
            "adapter_action": "md_campaign.prepare_lammps_campaign",
            "artifacts": (),
            "adapter_blockers": (str(exc),),
        }
    return {
        "adapter_action": "md_campaign.prepare_lammps_campaign",
        "artifacts": ("md_campaign_manifest.json", "md_physics_gate_report.json"),
        "md_campaign_plan": md_campaign_plan_payload(plan),
        "md_incident_count": _number(payload, "md_incident_count", 0.0),
    }


def ml_skill_adapter(payload: JsonMap) -> JsonMap:
    gate = payload.get("surrogate_training_gate")
    accepted = isinstance(gate, dict) and gate.get("accepted") is True
    return {
        "adapter_action": "ml_surrogate.assess_training_gate",
        "artifacts": ("surrogate_model_manifest.json", "surrogate_training_gate_report.json"),
        "surrogate_gate_accepted": accepted,
        "md_events_path": _text(payload, "md_events_path", ""),
    }


def feature_scale_skill_adapter(payload: JsonMap) -> JsonMap:
    return {
        "adapter_action": "level_set.prepare_transport_evolution",
        "artifacts": ("profile_timeline.json", "click_diagnostics.json"),
        "geometry_path": _text(payload, "geometry_path", ""),
        "iedf": _text(payload, "iedf", ""),
        "iadf": _text(payload, "iadf", ""),
        "process_time_s": _number(payload, "process_time_s", 0.0),
    }


def research_skill_adapter(payload: JsonMap) -> JsonMap:
    mode = _text(payload, "graphdb_mode", "dry_run")
    return {
        "adapter_action": "knowledge.prepare_graph_import_bundle",
        "artifacts": ("source_catalog.json", "graph_import_plan.json"),
        "research_question": _text(payload, "research_question", ""),
        "graphdb_mode": mode,
        "neo4j_write_enabled": mode == "write",
    }


def qa_skill_adapter(payload: JsonMap) -> JsonMap:
    quality_gates = _string_tuple(payload.get("quality_gates"))
    return {
        "adapter_action": "agent_run_quality.evaluate_gate_bundle",
        "artifacts": ("qa_report.json", "production_readiness_report.json"),
        "qa_status": "ready_for_runtime_review" if quality_gates else "quality_gates_missing",
        "quality_gates": quality_gates,
        "agent_run_ledger": _text(payload, "agent_run_ledger", ""),
    }


def _text(payload: JsonMap, field: str, default: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return default


def _number(payload: JsonMap, field: str, default: float) -> float:
    value = payload.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _range(payload: JsonMap, field: str, default: tuple[float, float]) -> tuple[float, float]:
    value = payload.get(field)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        first, second = value
        if isinstance(first, int | float) and isinstance(second, int | float):
            return float(first), float(second)
    return default


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))
