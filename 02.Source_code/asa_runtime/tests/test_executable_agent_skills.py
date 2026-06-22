from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime import SkillInvocationResult
from sim_agent.schemas._parse import JsonMap


def test_registered_agent_skills_execute_domain_adapters_and_write_artifacts(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_registered_agent_skills

    payload = _executable_skill_payload()

    invocations = run_registered_agent_skills(payload, output_dir=tmp_path)

    assert len(invocations) == 6
    for invocation in invocations:
        artifact_path = tmp_path / invocation.artifact_ref
        assert artifact_path.is_file()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["skill_id"] == invocation.skill_id
        assert artifact["status"] == "ready"
        assert artifact["result"]["adapter_invoked"] is True
        assert artifact["result"]["adapter_output"]["artifacts"]
    md_invocation = _by_skill(invocations, "prepare_and_verify_lammps_md")
    md_artifact = json.loads((tmp_path / md_invocation.artifact_ref).read_text(encoding="utf-8"))
    md_plan = md_artifact["result"]["adapter_output"]["md_campaign_plan"]
    assert md_plan["material_id"] == "Si"
    assert md_plan["ion_species"] == "Ar"
    assert md_plan["phases"] == ["amorphous"]
    assert md_plan["protocol_id"] == "continuous_stratified_bombardment"
    qa_invocation = _by_skill(invocations, "qa_physics_and_runtime_evidence")
    qa_artifact = json.loads((tmp_path / qa_invocation.artifact_ref).read_text(encoding="utf-8"))
    assert qa_artifact["result"]["adapter_output"]["qa_status"] == "ready_for_runtime_review"


def test_agents_sdk_runtime_dry_run_persists_executable_skill_artifacts(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run
    from sim_agent.llm_endpoints import ModelProviderConfig

    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "local_gateway",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "http://local-gateway.test/v1",
            "auth_mode": "none",
        }
    )

    result = run_agents_sdk_runtime_dry_run(_executable_skill_payload(), endpoint, output_dir=tmp_path)

    assert result.skill_invocations
    for invocation in result.skill_invocations:
        assert (tmp_path / invocation.artifact_ref).is_file()


def _by_skill(invocations: tuple[SkillInvocationResult, ...], skill_id: str) -> SkillInvocationResult:
    for invocation in invocations:
        if getattr(invocation, "skill_id") == skill_id:
            return invocation
    raise AssertionError(skill_id)


def _executable_skill_payload() -> JsonMap:
    return {
        "request_id": "executable-skills",
        "user_goal": "Plan Ar etching on amorphous Si and write skill artifacts",
        "material": "Si",
        "phase": "amorphous",
        "ion": "Ar",
        "md_incident_count": 500,
        "energy_range_eV": (30.0, 150.0),
        "polar_range_deg": (0.0, 55.0),
        "azimuth_range_deg": (0.0, 360.0),
        "md_events_path": "fixtures/md_events/md_events_small.jsonl",
        "surrogate_training_gate": {"accepted": True, "decision": "accepted_for_feature_scale"},
        "geometry_path": "fixtures/geometry/pr_hole.stl",
        "iedf": "histogram:30-150eV",
        "iadf": "uniform:0-55deg",
        "process_time_s": 1.0,
        "research_question": "Which force-field protocol supports Ar on Si etching?",
        "graphdb_mode": "dry_run",
        "agent_run_ledger": "agent_run_ledger.json",
        "quality_gates": ("md_physics_gate", "surrogate_training_gate", "level_set_profile_timeline"),
    }
