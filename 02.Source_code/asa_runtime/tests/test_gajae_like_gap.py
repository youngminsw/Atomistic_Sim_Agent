from __future__ import annotations

from pathlib import Path

from sim_agent.agents_sdk_runtime.spine_contract import runtime_spine_contract


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_gajae_like_runtime_contract_marks_all_spines_closed() -> None:
    contract = runtime_spine_contract()
    spines = {spine.spine_id: spine for spine in contract.spines}
    gap_text = "\n".join(spine.current_gap for spine in contract.spines)

    assert spines["provider_transport"].status.value == "complete"
    assert "protocol-specific endpoints" in spines["provider_transport"].current_gap
    assert all(spine.status.value == "complete" for spine in contract.spines)
    assert "persistent per-domain agent sessions" in gap_text
    assert "model/tool/runtime events" in gap_text
    assert "bounded planner, architect, critic, and executor subagent jobs" in gap_text


def test_gajae_like_gap_current_runtime_reports_gap_or_closed_state() -> None:
    from scripts.audit_runtime_spines import audit_runtime_spines

    audit = audit_runtime_spines(SOURCE_ROOT)
    spines = audit["spines"]

    assert spines["provider_transport"]["status"] in {"gap_open", "complete"}
    assert spines["agent_session"]["status"] in {"gap_open", "complete"}
    assert spines["agent_loop"]["status"] in {"gap_open", "complete"}
    assert spines["subagent_runtime"]["status"] in {"gap_open", "complete"}


def test_gajae_like_gap_detectors_are_connected_to_runtime_sources() -> None:
    agent_loop = (SOURCE_ROOT / "sim_agent" / "agents_sdk_runtime" / "agent_loop.py").read_text(encoding="utf-8")
    provider_model = (
        SOURCE_ROOT / "sim_agent" / "agents_sdk_runtime" / "provider_tool_choice_model.py"
    ).read_text(encoding="utf-8")
    agent_loop_contract = (
        SOURCE_ROOT / "sim_agent" / "agents_sdk_runtime" / "agent_loop_contract.py"
    ).read_text(encoding="utf-8")
    live_turn = (SOURCE_ROOT / "sim_agent" / "agent_runtime" / "live_agent_turn.py").read_text(encoding="utf-8")

    assert "class AsaAgentSession" in agent_loop_contract
    assert "ToolChoiceModel" in agent_loop
    assert "session.endpoint" in provider_model
    assert "tool_registry_for_agent" in live_turn
