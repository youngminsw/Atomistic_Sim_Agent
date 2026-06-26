from __future__ import annotations

from pathlib import Path


def test_workflow_gate_response_rejects_invalid_enum_accepts_valid_once(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "gate-protocol",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "qa_agent",
            "goal_id": "goal-gate-protocol",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {"gate_id": "approval", "gate_kind": "enum", "allowed_values": ["approve", "revise"]},
        },
        tmp_path,
    )

    denied = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "peer_agent", "value": "approve"},
    )
    invalid = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "qa_agent", "value": "maybe"},
    )
    valid = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "qa_agent", "value": "approve"},
    )
    duplicate = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "approval", "responder_agent_id": "qa_agent", "value": "approve"},
    )

    assert denied.status == "blocked"
    assert denied.blockers == ("workflow_gate_responder_denied",)
    assert invalid.status == "blocked"
    assert invalid.blockers == ("workflow_gate_invalid_enum_value",)
    assert valid.status == "accepted"
    assert valid.answered_at
    assert duplicate.status == "accepted"
    assert duplicate.blockers == ("workflow_gate_already_answered",)


def test_workflow_gate_response_rejects_unknown_gate_and_malformed_payload(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate

    malformed = respond_workflow_gate(tmp_path, {"workflow_id": "ralplan", "gate_id": "approval"})
    unknown = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "missing", "responder_agent_id": "qa_agent", "value": "approve"},
    )

    assert malformed.status == "blocked"
    assert malformed.blockers == ("workflow_gate_malformed_response",)
    assert unknown.status == "blocked"
    assert unknown.blockers == ("workflow_gate_unknown",)


def test_workflow_gate_response_schema_rejects_shape_mismatch_and_accepts_valid_object(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    run_workflow_harness_smoke(
        "ralplan",
        {
            "request_id": "schema-gate",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "research_agent",
            "goal_id": "goal-schema-gate",
            "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            "gate": {
                "gate_id": "clarify",
                "gate_kind": "response_schema",
                "response_schema": {
                    "type": "object",
                    "required": ["decision"],
                    "properties": {"decision": {"type": "string"}},
                },
            },
        },
        tmp_path,
    )

    scalar = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "clarify", "responder_agent_id": "research_agent", "value": "clear"},
    )
    missing_required = respond_workflow_gate(
        tmp_path,
        {"workflow_id": "ralplan", "gate_id": "clarify", "responder_agent_id": "research_agent", "value": {}},
    )
    wrong_nested_type = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "clarify",
            "responder_agent_id": "research_agent",
            "value": {"decision": 1},
        },
    )
    valid = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "clarify",
            "responder_agent_id": "research_agent",
            "value": {"decision": "clear"},
        },
    )

    assert scalar.status == "blocked"
    assert scalar.blockers == ("workflow_gate_response_schema_mismatch",)
    assert missing_required.status == "blocked"
    assert missing_required.blockers == ("workflow_gate_response_schema_mismatch",)
    assert wrong_nested_type.status == "blocked"
    assert wrong_nested_type.blockers == ("workflow_gate_response_schema_mismatch",)
    assert valid.status == "accepted"
