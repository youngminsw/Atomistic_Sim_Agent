from __future__ import annotations

from sim_agent.agents_sdk_runtime.spine_contract import (
    RUNTIME_SPINE_CONTRACT_VERSION,
    RuntimeSpineStatus,
    runtime_spine_contract,
    runtime_spine_matrix,
)


def test_runtime_spine_contract_names_all_eight_runtime_spines() -> None:
    contract = runtime_spine_contract()

    assert RUNTIME_SPINE_CONTRACT_VERSION == "asa_runtime_spine_contract_v1"
    assert tuple(spine.name for spine in contract.spines) == (
        "Model/Provider/Transport",
        "AgentSession",
        "AgentLoop",
        "Prompt/Skill/Workflow Assembly",
        "Subagent/Task Runtime",
        "Context/Compaction/Resume",
        "Tool Registry/Tool Runtime",
        "TUI/UX/Observability",
    )


def test_runtime_spine_contract_records_assertion_and_current_gap_for_each_spine() -> None:
    contract = runtime_spine_contract()

    for spine in contract.spines:
        assert spine.status is RuntimeSpineStatus.GAP_OPEN
        assert spine.required_assertion
        assert spine.current_gap
        assert spine.acceptance_probe
        assert spine.evidence_path == ".omo/evidence/task-1-asa-runtime-spine-gap-closure.json"
        assert spine.doc_anchor.startswith("#")


def test_runtime_spine_matrix_is_json_ready_and_keyed_by_stable_spine_id() -> None:
    matrix = runtime_spine_matrix()

    assert set(matrix) == {
        "provider_transport",
        "agent_session",
        "agent_loop",
        "assembly",
        "subagent_runtime",
        "context_resume",
        "tool_runtime",
        "tui_observability",
    }
    assert matrix["provider_transport"]["status"] == "gap_open"
    assert "/v1/responses" in matrix["provider_transport"]["current_gap"]
