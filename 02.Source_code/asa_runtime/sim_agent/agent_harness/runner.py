from __future__ import annotations

from sim_agent.input_planner import InputPlanningResult, plan_simulation_input
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.md_campaign import MDCampaignPlan, plan_md_campaign
from sim_agent.schemas._parse import JsonMap
from sim_agent.schemas.request import EtchRecipe, RunArtifact, SimulationRequest
from sim_agent.schemas.state import SimulationScene

from .client import OfflineModelClient
from .tools import ToolRegistry, default_tool_registry
from .types import AgentRunResult, ClarificationRequired, RunStatus, ToolTraceEvent


class SimulationAgentHarness:
    def __init__(
        self,
        endpoint: ModelProviderConfig,
        client: OfflineModelClient,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._client = client
        self._tool_registry = tool_registry or default_tool_registry()

    def plan(self, payload: JsonMap) -> AgentRunResult:
        request_id = _request_id(payload)
        if _requested_fake_completion(payload):
            return AgentRunResult(
                run_id=f"plan-{request_id}",
                status=RunStatus.BLOCKED,
                final_output="verification_missing",
                clarification=None,
                md_campaign_plan=None,
                artifacts=(),
                trace=(ToolTraceEvent("inspect_artifact_manifest", "no artifacts supplied"),),
                verification_evidence=(),
            )

        input_plan = plan_simulation_input(payload)
        if input_plan.missing_fields:
            return AgentRunResult(
                run_id=f"plan-{request_id}",
                status=RunStatus.CLARIFICATION_REQUIRED,
                final_output=_clarification_output(input_plan),
                clarification=ClarificationRequired(
                    missing_fields=input_plan.missing_fields,
                    question=_clarification_question(input_plan),
                ),
                artifacts=(),
                md_campaign_plan=None,
                trace=_clarification_trace(input_plan),
                verification_evidence=("input_planner",),
            )

        request = SimulationRequest.from_mapping(payload)
        plan_text = self._client.plan(
            controller_name="simulation-controller",
            model_spec=self._endpoint.to_agents_sdk_model_spec(),
            registry=self._tool_registry,
        )
        run_id = f"plan-{request.request_id}"
        campaign = _campaign_from_request(request)
        artifacts = (
            RunArtifact(
                f"{run_id}:md_campaign_plan",
                f"evidence/{request.request_id}/md_campaign_plan.json",
                "md_campaign_plan",
            ),
            RunArtifact(f"{run_id}:manifest", f"evidence/{request.request_id}/manifest.json", "run_manifest"),
            RunArtifact(
                f"{run_id}:validated_request",
                f"evidence/{request.request_id}/validated_request.json",
                "validated_request",
            ),
        )
        return AgentRunResult(
            run_id=run_id,
            status=RunStatus.PLANNED,
            final_output=f"planned:{plan_text}",
            clarification=None,
            md_campaign_plan=campaign,
            artifacts=artifacts,
            trace=_planning_trace(input_plan) + (
                ToolTraceEvent("plan_md_campaign", _campaign_summary(campaign)),
                ToolTraceEvent("validate_simulation_request", request.request_id),
                ToolTraceEvent("create_artifact_manifest", "md_campaign_plan,run_manifest,validated_request"),
                ToolTraceEvent("record_run_trace", run_id),
            ),
            verification_evidence=_verification_evidence(input_plan) + _campaign_evidence(campaign),
        )


def _request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def _requested_fake_completion(payload: JsonMap) -> bool:
    return payload.get("requested_state") == "complete_without_artifacts"


def _clarification_output(input_plan: InputPlanningResult) -> str:
    output = f"clarification_required: {_clarification_question(input_plan)}"
    if input_plan.model_training_required:
        return f"{output} model_training_required=true training_reason={input_plan.training_reason}"
    return output


def _clarification_question(input_plan: InputPlanningResult) -> str:
    return " ".join(prompt.question for prompt in input_plan.clarifications)


def _clarification_trace(input_plan: InputPlanningResult) -> tuple[ToolTraceEvent, ...]:
    trace = [
        ToolTraceEvent("inspect_request_inputs", "physical inputs incomplete"),
        ToolTraceEvent("plan_simulation_input", _planning_summary(input_plan)),
    ]
    if input_plan.model_training_required:
        trace.append(ToolTraceEvent("mark_surrogate_training_required", input_plan.training_reason))
    return tuple(trace)


def _planning_trace(input_plan: InputPlanningResult) -> tuple[ToolTraceEvent, ...]:
    trace = [ToolTraceEvent("plan_simulation_input", _planning_summary(input_plan))]
    if input_plan.trained_kernel_id:
        trace.append(ToolTraceEvent("select_surrogate_kernel", input_plan.trained_kernel_id))
    if input_plan.model_training_required:
        trace.append(ToolTraceEvent("mark_surrogate_training_required", input_plan.training_reason))
    return tuple(trace)


def _planning_summary(input_plan: InputPlanningResult) -> str:
    if input_plan.trained_kernel_id:
        return f"{input_plan.mode}:{input_plan.feature_type}:kernel={input_plan.trained_kernel_id}"
    if input_plan.model_training_required:
        return f"{input_plan.mode}:{input_plan.feature_type}:training_required={input_plan.training_reason}"
    return f"{input_plan.mode}:{input_plan.feature_type}:missing={','.join(input_plan.missing_fields)}"


def _verification_evidence(input_plan: InputPlanningResult) -> tuple[str, ...]:
    evidence = ["input_planner", "validated_request", "artifact_manifest"]
    if input_plan.trained_kernel_id:
        evidence.append(f"surrogate_kernel:{input_plan.trained_kernel_id}")
    if input_plan.model_training_required:
        evidence.append(f"surrogate_training_required:{input_plan.training_reason}")
    return tuple(evidence)


def _campaign_from_request(request: SimulationRequest) -> MDCampaignPlan:
    angular = request.recipe.ion_angular_distribution
    return plan_md_campaign(
        material_id=request.scene.surface_state.material_id,
        ion_species=request.recipe.ion_species,
        phases=_target_phases(request.scene),
        energy_range_eV=_energy_range(request.recipe),
        polar_range_deg=(angular.polar_min_deg, angular.polar_max_deg),
        azimuth_range_deg=(angular.azimuth_min_deg, angular.azimuth_max_deg),
        active_layer_thickness_nm=request.scene.surface_state.active_layer_thickness_nm,
    )


def _energy_range(recipe: EtchRecipe) -> tuple[float, float]:
    bins = recipe.ion_energy_distribution.bins
    if not bins:
        raise ValueError("energy_distribution_bins_required")
    return (min(bin.min for bin in bins), max(bin.max for bin in bins))


def _target_phases(scene: SimulationScene) -> tuple[str, ...]:
    phases: list[str] = []
    target_material_id = scene.surface_state.material_id
    for material in scene.material_stack.materials:
        if material.role == "target" and material.material_id == target_material_id:
            _append_unique(phases, material.phase)
    for volume_state in scene.volume_states:
        if volume_state.material_id == target_material_id:
            _append_unique(phases, volume_state.phase)
    _append_unique(phases, scene.surface_state.phase)
    return tuple(phases)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _campaign_summary(campaign: MDCampaignPlan) -> str:
    energy = campaign.energy_strata
    polar = campaign.polar_strata
    return (
        f"{campaign.material_id}/{campaign.ion_species}:"
        f"{','.join(campaign.phases)}:"
        f"{energy.minimum}-{energy.maximum}eV:"
        f"{polar.minimum}-{polar.maximum}deg"
    )


def _campaign_evidence(campaign: MDCampaignPlan) -> tuple[str, ...]:
    return (
        f"md_campaign:{campaign.protocol_id}",
        f"md_campaign_phases:{','.join(campaign.phases)}",
        f"md_campaign_energy_eV:{campaign.energy_strata.minimum}:{campaign.energy_strata.maximum}",
        f"md_campaign_layer_renewal_nm:{campaign.layer_renewal.removed_depth_threshold_nm}",
    )
