from __future__ import annotations

import json

import pytest

from sim_agent.agents_sdk_runtime.runtime_events import (
    REDACTED_VALUE,
    RuntimeEventOrderingError,
    project_runtime_events,
    runtime_event_from_json,
    runtime_event_json_line,
    runtime_event_to_json,
    validate_runtime_event_order,
)
from sim_agent.agents_sdk_runtime.types import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    RuntimeEventType,
)


def test_runtime_event_schema_defines_normalized_event_types() -> None:
    event = RuntimeEvent(
        sequence=1,
        at=1.25,
        session_id="session-1",
        turn_id="turn-1",
        event_type=RuntimeEventType.MODEL_START,
        agent_id="orchestrator",
        payload={"model": "gpt-5.5", "status": "running"},
    )

    encoded = runtime_event_to_json(event)
    decoded = runtime_event_from_json(encoded)

    assert decoded == event
    assert encoded["schema_version"] == RUNTIME_EVENT_SCHEMA_VERSION
    assert {kind.value for kind in RuntimeEventType} == {
        "model_start",
        "model_delta",
        "model_end",
        "tool_start",
        "tool_end",
        "message_append",
        "subagent_status",
        "compaction_checkpoint",
        "workflow_gate",
        "blocker",
        "resume",
        "cancellation",
    }


def test_runtime_event_redaction_removes_sensitive_payload_values() -> None:
    event = RuntimeEvent(
        sequence=1,
        at=1.0,
        session_id="session-1",
        turn_id="turn-1",
        event_type=RuntimeEventType.TOOL_START,
        agent_id="orchestrator",
        payload={
            "tool_name": "gateway_call",
            "api_key": "sk-live",
            "nested": {"refresh_token": "oauth-token", "safe": "kept"},
            "headers": [{"Authorization": "Bearer secret"}],
        },
    )

    encoded = runtime_event_to_json(event)
    payload = encoded["payload"]

    assert isinstance(payload, dict)
    assert payload["api_key"] == REDACTED_VALUE
    assert payload["nested"] == {"refresh_token": REDACTED_VALUE, "safe": "kept"}
    assert payload["headers"] == [{"Authorization": REDACTED_VALUE}]
    assert "sk-live" not in runtime_event_json_line(event)


def test_runtime_event_ordering_accepts_streaming_turn_with_tool_span() -> None:
    events = (
        _event(1, RuntimeEventType.RESUME, {"summary": "restored session"}),
        _event(2, RuntimeEventType.MODEL_START, {"model": "gpt-5.5"}),
        _event(3, RuntimeEventType.MODEL_DELTA, {"text": "Planning"}),
        _event(4, RuntimeEventType.TOOL_START, {"tool_call_id": "tool-1", "tool_name": "md_plan"}),
        _event(5, RuntimeEventType.TOOL_END, {"tool_call_id": "tool-1", "status": "ok"}),
        _event(6, RuntimeEventType.MESSAGE_APPEND, {"text": "MD plan ready"}),
        _event(7, RuntimeEventType.SUBAGENT_STATUS, {"target": "md_agent", "status": "idle"}),
        _event(8, RuntimeEventType.COMPACTION_CHECKPOINT, {"summary": "checkpoint written"}),
        _event(9, RuntimeEventType.WORKFLOW_GATE, {"target": "qa_gate", "status": "pass"}),
        _event(10, RuntimeEventType.MODEL_END, {"status": "ok"}),
        _event(11, RuntimeEventType.CANCELLATION, {"status": "cancelled"}),
    )

    ordered = validate_runtime_event_order(events)

    assert ordered == events


def test_runtime_event_projection_for_fake_streaming_turn() -> None:
    events = (
        _event(1, RuntimeEventType.MODEL_START, {"model": "gpt-5.5"}),
        _event(2, RuntimeEventType.MODEL_DELTA, {"text": "Streaming profile plan"}),
        _event(3, RuntimeEventType.BLOCKER, {"summary": "QA gate required", "status": "blocked"}),
        _event(4, RuntimeEventType.MODEL_END, {"status": "ok"}),
    )

    projections = project_runtime_events(events)

    assert [projection.label for projection in projections] == [
        "orchestrator model started",
        "orchestrator streaming",
        "blocker",
        "orchestrator model completed",
    ]
    assert projections[1].detail == "Streaming profile plan"
    assert projections[2].tone == "warning"


def test_runtime_event_ordering_rejects_out_of_order_tool_end() -> None:
    events = (
        _event(1, RuntimeEventType.MODEL_START, {"model": "gpt-5.5"}),
        _event(2, RuntimeEventType.TOOL_END, {"tool_call_id": "tool-1", "status": "ok"}),
    )

    with pytest.raises(RuntimeEventOrderingError, match="tool_end requires a matching active tool_start"):
        validate_runtime_event_order(events)


def test_runtime_event_json_line_round_trips() -> None:
    event = _event(1, RuntimeEventType.MESSAGE_APPEND, {"text": "hello"})

    decoded = runtime_event_from_json(json.loads(runtime_event_json_line(event)))

    assert decoded == event


def _event(sequence: int, event_type: RuntimeEventType, payload: dict[str, object]) -> RuntimeEvent:
    return RuntimeEvent(
        sequence=sequence,
        at=float(sequence),
        session_id="session-1",
        turn_id="turn-1",
        event_type=event_type,
        agent_id="orchestrator",
        payload=payload,
    )
