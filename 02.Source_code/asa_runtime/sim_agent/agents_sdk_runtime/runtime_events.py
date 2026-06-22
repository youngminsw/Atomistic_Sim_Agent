from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require

from .types import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    RuntimeEventProjection,
    RuntimeEventType,
)


REDACTED_VALUE: Final = "[redacted]"
SENSITIVE_KEY_PARTS: Final = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "refresh_token",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class RuntimeEventOrderingError(Exception):
    event: RuntimeEvent
    reason: str

    def __str__(self) -> str:
        return f"runtime event {self.event.sequence} {self.event.event_type.value}: {self.reason}"


def runtime_event_from_json(value: JsonMap) -> RuntimeEvent:
    schema_version = as_str(require(value, "schema_version"), "schema_version")
    if schema_version != RUNTIME_EVENT_SCHEMA_VERSION:
        raise RuntimeEventOrderingError(_invalid_event(), f"unsupported schema version {schema_version}")
    return RuntimeEvent(
        sequence=_int_field(value, "sequence"),
        at=_float_field(value, "at"),
        session_id=as_str(require(value, "session_id"), "session_id"),
        turn_id=as_str(require(value, "turn_id"), "turn_id"),
        event_type=_event_type(as_str(require(value, "event_type"), "event_type")),
        agent_id=as_str(require(value, "agent_id"), "agent_id"),
        payload=as_mapping(require(value, "payload"), "payload"),
        correlation_id=_optional_text(value, "correlation_id"),
        parent_id=_optional_text(value, "parent_id"),
    )


def runtime_event_to_json(event: RuntimeEvent, *, redact: bool = True) -> JsonMap:
    return RuntimeEvent(
        sequence=event.sequence,
        at=event.at,
        session_id=event.session_id,
        turn_id=event.turn_id,
        event_type=event.event_type,
        agent_id=event.agent_id,
        payload=redact_json_map(event.payload) if redact else event.payload,
        correlation_id=event.correlation_id,
        parent_id=event.parent_id,
    ).to_json()


def runtime_event_json_line(event: RuntimeEvent, *, redact: bool = True) -> str:
    return json.dumps(runtime_event_to_json(event, redact=redact), sort_keys=True) + "\n"


def validate_runtime_event_order(events: Sequence[RuntimeEvent]) -> tuple[RuntimeEvent, ...]:
    active_models: set[str] = set()
    active_tools: set[str] = set()
    previous_sequence = -1
    for event in events:
        if event.sequence <= previous_sequence:
            raise RuntimeEventOrderingError(event, "sequence must increase monotonically")
        previous_sequence = event.sequence
        _validate_span_event(event, active_models, active_tools)
    return tuple(events)


def redact_json_map(payload: JsonMap) -> JsonMap:
    return {key: _redact_value(key, value) for key, value in payload.items()}


def project_runtime_event(event: RuntimeEvent) -> RuntimeEventProjection:
    status = _payload_text(event.payload, "status", _default_status(event.event_type))
    detail = _payload_text(event.payload, "summary", _payload_text(event.payload, "text", ""))
    return RuntimeEventProjection(
        sequence=event.sequence,
        event_type=event.event_type,
        agent_id=event.agent_id,
        status=status,
        label=_event_label(event),
        detail=detail,
        tone=_event_tone(event.event_type, status),
    )


def project_runtime_events(events: Sequence[RuntimeEvent]) -> tuple[RuntimeEventProjection, ...]:
    return tuple(project_runtime_event(event) for event in validate_runtime_event_order(events))


def _validate_span_event(event: RuntimeEvent, active_models: set[str], active_tools: set[str]) -> None:
    match event.event_type:
        case RuntimeEventType.MODEL_START:
            active_models.add(event.turn_id)
        case RuntimeEventType.MODEL_DELTA:
            if event.turn_id not in active_models:
                raise RuntimeEventOrderingError(event, "model_delta requires an active model_start")
        case RuntimeEventType.MODEL_END:
            if event.turn_id not in active_models:
                raise RuntimeEventOrderingError(event, "model_end requires an active model_start")
            active_models.remove(event.turn_id)
        case RuntimeEventType.TOOL_START:
            active_tools.add(_tool_key(event))
        case RuntimeEventType.TOOL_END:
            tool_key = _tool_key(event)
            if tool_key not in active_tools:
                raise RuntimeEventOrderingError(event, "tool_end requires a matching active tool_start")
            active_tools.remove(tool_key)
        case (
            RuntimeEventType.MESSAGE_APPEND
            | RuntimeEventType.SUBAGENT_STATUS
            | RuntimeEventType.COMPACTION_CHECKPOINT
            | RuntimeEventType.WORKFLOW_GATE
            | RuntimeEventType.BLOCKER
            | RuntimeEventType.RESUME
            | RuntimeEventType.CANCELLATION
        ):
            return
        case unreachable:
            assert_never(unreachable)


def _redact_value(key: str, value: object) -> object:
    if _sensitive_key(key):
        return REDACTED_VALUE
    if isinstance(value, dict):
        return {str(child_key): _redact_value(str(child_key), child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item) for item in value)
    return value


def _sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _tool_key(event: RuntimeEvent) -> str:
    tool_call_id = event.payload.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id
    if event.correlation_id:
        return event.correlation_id
    tool_name = event.payload.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return f"{event.turn_id}:{event.agent_id}:{tool_name}"
    return f"{event.turn_id}:{event.agent_id}"


def _event_label(event: RuntimeEvent) -> str:
    target = _payload_text(event.payload, "tool_name", _payload_text(event.payload, "target", event.agent_id))
    match event.event_type:
        case RuntimeEventType.MODEL_START:
            return f"{event.agent_id} model started"
        case RuntimeEventType.MODEL_DELTA:
            return f"{event.agent_id} streaming"
        case RuntimeEventType.MODEL_END:
            return f"{event.agent_id} model completed"
        case RuntimeEventType.TOOL_START:
            return f"{target} tool started"
        case RuntimeEventType.TOOL_END:
            return f"{target} tool completed"
        case RuntimeEventType.MESSAGE_APPEND:
            return f"{event.agent_id} message"
        case RuntimeEventType.SUBAGENT_STATUS:
            return f"{target} subagent"
        case RuntimeEventType.COMPACTION_CHECKPOINT:
            return "compaction checkpoint"
        case RuntimeEventType.WORKFLOW_GATE:
            return f"{target} workflow gate"
        case RuntimeEventType.BLOCKER:
            return "blocker"
        case RuntimeEventType.RESUME:
            return "resume"
        case RuntimeEventType.CANCELLATION:
            return "cancellation"
        case unreachable:
            assert_never(unreachable)


def _event_tone(event_type: RuntimeEventType, status: str) -> str:
    if status in {"blocked", "failed", "error"}:
        return "warning"
    if event_type in {RuntimeEventType.BLOCKER, RuntimeEventType.CANCELLATION}:
        return "warning"
    if event_type in {RuntimeEventType.WORKFLOW_GATE, RuntimeEventType.COMPACTION_CHECKPOINT}:
        return "accent"
    return "body"


def _default_status(event_type: RuntimeEventType) -> str:
    return {
        RuntimeEventType.MODEL_START: "running",
        RuntimeEventType.MODEL_DELTA: "streaming",
        RuntimeEventType.MODEL_END: "ok",
        RuntimeEventType.TOOL_START: "running",
        RuntimeEventType.TOOL_END: "ok",
        RuntimeEventType.MESSAGE_APPEND: "ok",
        RuntimeEventType.SUBAGENT_STATUS: "active",
        RuntimeEventType.COMPACTION_CHECKPOINT: "checkpoint",
        RuntimeEventType.WORKFLOW_GATE: "gate",
        RuntimeEventType.BLOCKER: "blocked",
        RuntimeEventType.RESUME: "resumed",
        RuntimeEventType.CANCELLATION: "cancelled",
    }[event_type]


def _payload_text(payload: JsonMap, field: str, default: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) and value else default


def _event_type(value: str) -> RuntimeEventType:
    try:
        return RuntimeEventType(value)
    except ValueError as exc:
        raise RuntimeEventOrderingError(_invalid_event(), f"unknown event type {value}") from exc


def _int_field(mapping: JsonMap, field: str) -> int:
    value = require(mapping, field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeEventOrderingError(_invalid_event(), f"{field} must be an integer")


def _float_field(mapping: JsonMap, field: str) -> float:
    value = require(mapping, field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeEventOrderingError(_invalid_event(), f"{field} must be a number")


def _optional_text(mapping: JsonMap, field: str) -> str:
    value = mapping.get(field)
    return "" if value is None else as_str(value, field)


def _invalid_event() -> RuntimeEvent:
    return RuntimeEvent(-1, 0.0, "invalid", "invalid", RuntimeEventType.BLOCKER, "runtime_event_parser", {})
