from __future__ import annotations

import json
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


def test_agent_plan_http_response_includes_md_campaign_payload() -> None:
    from sim_agent.ui.agent_plan import build_agent_plan_http_response

    body, status_code = build_agent_plan_http_response(_load_request("valid_ar_si_pr_hole.json"))
    campaign = as_mapping(body["md_campaign_plan"], "md_campaign_plan")
    team = as_mapping(body["team_session_contract"], "team_session_contract")
    energy = as_mapping(campaign["energy_strata"], "energy_strata")
    polar = as_mapping(campaign["polar_strata"], "polar_strata")
    layer = as_mapping(campaign["layer_renewal"], "layer_renewal")

    assert status_code == 200
    assert body["status"] == "planned"
    assert team["contract_version"] == "agent_team_session_contract_v1"
    assert team["heartbeat_interval_s"] == 3600
    assert team["inter_agent_call_timeout_s"] == 1800
    assert team["qa_gates"]["slurm_job_script"] == "qa_before_submit"
    assert set(team["call_matrix"]["orchestrator"]) == {
        "md_agent",
        "ml_agent",
        "feature_scale_agent",
        "research_agent",
        "qa_agent",
    }
    assert set(team["call_matrix"]["md_agent"]) == {
        "orchestrator",
        "research_agent",
        "qa_agent",
    }
    assert campaign["protocol_id"] == "continuous_stratified_bombardment"
    assert campaign["material_id"] == "Si"
    assert campaign["ion_species"] == "Ar"
    assert campaign["phases"] == ["crystal"]
    assert energy["axis"] == "energy_eV"
    assert energy["minimum"] == 20.0
    assert energy["maximum"] == 200.0
    assert polar["axis"] == "polar_deg"
    assert polar["minimum"] == 0.0
    assert polar["maximum"] == 60.0
    assert "rdf_order_features" in campaign["pre_state_descriptors"]
    assert layer["renewal_action"] == "expose_next_volume_state"
    assert layer["removed_depth_threshold_nm"] == 1.0
