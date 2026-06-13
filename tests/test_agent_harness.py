from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def test_missing_physical_inputs_return_typed_clarification() -> None:
    from sim_agent.agent_harness import OfflineModelClient, RunStatus, SimulationAgentHarness
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("missing_recipe.json")
    endpoint = ModelProviderConfig.from_mapping(payload["llm_endpoint"])
    harness = SimulationAgentHarness(endpoint=endpoint, client=OfflineModelClient())

    result = harness.plan(payload)

    assert result.status == RunStatus.CLARIFICATION_REQUIRED
    assert result.clarification is not None
    assert result.clarification.missing_fields == (
        "geometry",
        "material",
        "phase",
        "ion_species",
        "iedf",
        "iadf",
        "flux",
    )
    assert "IonEnergyDistribution" in result.final_output
    assert result.trace[0].tool_name == "inspect_request_inputs"


def test_valid_request_records_tool_trace_and_artifact_manifest() -> None:
    from sim_agent.agent_harness import OfflineModelClient, RunStatus, SimulationAgentHarness
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    endpoint = ModelProviderConfig.from_mapping(payload["llm_endpoint"])
    client = OfflineModelClient()
    harness = SimulationAgentHarness(endpoint=endpoint, client=client)

    result = harness.plan(payload)

    assert result.status == RunStatus.PLANNED
    assert client.calls == ("plan:simulation-controller",)
    assert [event.tool_name for event in result.trace] == [
        "plan_simulation_input",
        "select_surrogate_kernel",
        "plan_md_campaign",
        "validate_simulation_request",
        "create_artifact_manifest",
        "record_run_trace",
    ]
    assert result.trace[1].summary == "Ar_on_Si__physical_v001"
    assert result.trace[2].summary == "Si/Ar:crystal:20.0-200.0eV:0.0-60.0deg"
    assert "md_campaign:continuous_stratified_bombardment" in result.verification_evidence
    assert "md_campaign_phases:crystal" in result.verification_evidence
    assert "surrogate_kernel:Ar_on_Si__physical_v001" in result.verification_evidence
    assert {artifact.artifact_type for artifact in result.artifacts} == {
        "md_campaign_plan",
        "run_manifest",
        "validated_request",
    }


def test_unknown_material_trace_keeps_training_required_while_asking_for_missing_inputs() -> None:
    from sim_agent.agent_harness import OfflineModelClient, RunStatus, SimulationAgentHarness
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("ar_on_unknown_material.json")
    endpoint = ModelProviderConfig.from_mapping(payload["llm_endpoint"])
    client = OfflineModelClient()
    harness = SimulationAgentHarness(endpoint=endpoint, client=client)

    result = harness.plan(payload)

    assert result.status == RunStatus.CLARIFICATION_REQUIRED
    assert client.calls == ()
    assert result.clarification is not None
    assert result.clarification.missing_fields == ("geometry", "material", "phase", "iedf", "iadf", "flux")
    assert "model_training_required=true" in result.final_output
    assert "no_trained_expert_for_Ar_on_UnobtaniumFixture" in result.final_output
    assert [event.tool_name for event in result.trace] == [
        "inspect_request_inputs",
        "plan_simulation_input",
        "mark_surrogate_training_required",
    ]


def test_completion_without_artifacts_is_blocked() -> None:
    from sim_agent.agent_harness import OfflineModelClient, RunStatus, SimulationAgentHarness
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("no_artifacts_complete_request.json")
    endpoint = ModelProviderConfig.from_mapping(payload["llm_endpoint"])
    harness = SimulationAgentHarness(endpoint=endpoint, client=OfflineModelClient())

    result = harness.plan(payload)

    assert result.status == RunStatus.BLOCKED
    assert result.final_output == "verification_missing"
    assert result.artifacts == ()
    assert result.verification_evidence == ()


def test_tool_registry_exposes_future_simulation_boundaries() -> None:
    from sim_agent.agent_harness import default_tool_registry

    registry = default_tool_registry()

    assert {
        "validate_simulation_request",
        "geometry_ingestion",
        "md_campaign_planning",
        "surrogate_status",
        "feature_transport",
        "level_set_evolution",
        "compute_routing",
        "artifact_manifest",
        "literature_registry",
        "ui_run_status",
    }.issubset(registry.tool_names)


def test_smoke_agent_plan_cli_reports_clarification() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agent_plan.py"),
            "--offline",
            "--request",
            str(REQUEST_ROOT / "missing_recipe.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "clarification_required=true" in result.stdout
    assert "IonEnergyDistribution" in result.stdout
    assert "IonAngularDistribution" in result.stdout
    assert "material" in result.stdout
    assert "phase" in result.stdout
    assert "geometry" in result.stdout


def test_smoke_agent_plan_cli_blocks_fake_completion() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agent_plan.py"),
            "--offline",
            "--request",
            str(REQUEST_ROOT / "no_artifacts_complete_request.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "verification_missing" in result.stdout
