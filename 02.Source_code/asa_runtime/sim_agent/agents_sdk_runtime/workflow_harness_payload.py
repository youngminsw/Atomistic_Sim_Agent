from __future__ import annotations

from sim_agent.schemas._parse import JsonMap


def request_id(payload: JsonMap) -> str:
    value = payload.get("request_id")
    if isinstance(value, str) and value:
        return value
    return "anonymous"


def text_value(value: object, fallback: str) -> str:
    if isinstance(value, str) and value:
        return value
    if value is None:
        return fallback
    return str(value)


def agent_id(payload: JsonMap, field: str, fallback: str) -> str:
    value = payload.get(field)
    if isinstance(value, str) and value:
        return value
    return fallback


def optional_gate(payload: JsonMap) -> JsonMap | None:
    gate = payload.get("gate")
    if isinstance(gate, dict):
        return gate
    return None


def evidence_keys(payload: JsonMap) -> tuple[str, ...]:
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        return tuple(sorted(key for key, value in evidence.items() if isinstance(key, str) and evidence_value_present(value)))
    evidence_ledger = payload.get("evidence_ledger")
    if isinstance(evidence_ledger, list | tuple):
        keys: set[str] = set()
        for item in evidence_ledger:
            if isinstance(item, str) and item:
                keys.add(item)
            elif isinstance(item, dict):
                key = item.get("key") or item.get("id") or item.get("name")
                if isinstance(key, str) and key:
                    keys.add(key)
        return tuple(sorted(keys))
    return tuple(
        sorted(key for key in payload if isinstance(key, str) and key not in {"request_id", "user_goal", "workflow_id"})
    )


def evidence_value_present(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | dict | set):
        return bool(value)
    return True
