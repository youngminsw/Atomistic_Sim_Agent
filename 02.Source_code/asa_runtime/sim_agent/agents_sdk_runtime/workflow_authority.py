from __future__ import annotations

from typing import Final

from sim_agent.schemas._parse import JsonMap


type WorkflowSchemaValue = str | int | float | bool | None | JsonMap | list["WorkflowSchemaValue"] | tuple[
    "WorkflowSchemaValue", ...
]

ORCHESTRATOR_AGENT_ID: Final = "orchestrator"


def workflow_authority_blocker(actor: str, owner: str, target: str) -> str:
    if actor == ORCHESTRATOR_AGENT_ID:
        if owner == ORCHESTRATOR_AGENT_ID or (owner == target and target != ORCHESTRATOR_AGENT_ID):
            return ""
        return "workflow_authority_peer_denied"
    if target == ORCHESTRATOR_AGENT_ID:
        return "workflow_authority_orchestrator_denied"
    if actor == owner and actor == target:
        return ""
    return "workflow_authority_peer_denied"


def response_schema_blocker(value: WorkflowSchemaValue, schema: JsonMap) -> str:
    schema_type = schema.get("type")
    if schema_type is not None and not matches_schema_type(value, schema_type):
        return "workflow_gate_response_schema_mismatch"
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return "workflow_gate_response_schema_mismatch"
    required = schema.get("required")
    if isinstance(required, list):
        if not isinstance(value, dict):
            return "workflow_gate_response_schema_mismatch"
        for field in required:
            if isinstance(field, str) and field not in value:
                return "workflow_gate_response_schema_mismatch"
    if schema.get("additionalProperties") is False and isinstance(value, dict):
        properties = schema.get("properties")
        allowed = {field for field in properties if isinstance(field, str)} if isinstance(properties, dict) else set()
        if any(not isinstance(field, str) or field not in allowed for field in value):
            return "workflow_gate_response_schema_mismatch"
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(value, dict):
        for field, field_schema in properties.items():
            if not isinstance(field, str) or field not in value or not isinstance(field_schema, dict):
                continue
            blocker = response_schema_blocker(value[field], field_schema)
            if blocker:
                return blocker
    if isinstance(value, list):
        blocker = _array_schema_blocker(value, schema)
        if blocker:
            return blocker
    return ""


def matches_schema_type(value: WorkflowSchemaValue, schema_type: WorkflowSchemaValue) -> bool:
    if isinstance(schema_type, list):
        return any(matches_schema_type(value, item) for item in schema_type)
    if not isinstance(schema_type, str):
        return True
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _array_schema_blocker(value: list[WorkflowSchemaValue], schema: JsonMap) -> str:
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        return "workflow_gate_response_schema_mismatch"
    if isinstance(max_items, int) and len(value) > max_items:
        return "workflow_gate_response_schema_mismatch"
    items_schema = schema.get("items")
    if isinstance(items_schema, dict):
        for item in value:
            blocker = response_schema_blocker(item, items_schema)
            if blocker:
                return blocker
    return ""
