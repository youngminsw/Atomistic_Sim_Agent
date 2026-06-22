from __future__ import annotations

from pathlib import Path

from sim_agent.agents_sdk_runtime.spine_contract import runtime_spine_contract


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_gajae_like_gap_contract_marks_provider_session_loop_and_subagent_blockers() -> None:
    contract = runtime_spine_contract()
    gap_text = "\n".join(spine.current_gap for spine in contract.spines)

    assert "fixed /v1/responses" in gap_text
    assert "frozen DTO" in gap_text
    assert "one-shot" in gap_text
    assert "detached controllable job runtime" in gap_text


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
    live_turn = (SOURCE_ROOT / "sim_agent" / "agent_runtime" / "live_agent_turn.py").read_text(encoding="utf-8")

    assert "class AsaAgentSession" in agent_loop
    assert "ToolChoiceModel" in agent_loop
    assert "session.endpoint" in provider_model
    assert "StaticToolChoiceModel" in live_turn or "offline" in live_turn
