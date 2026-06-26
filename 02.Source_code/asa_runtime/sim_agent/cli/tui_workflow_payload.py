from __future__ import annotations

import json

from sim_agent.schemas._parse import JsonMap


JsonValue = JsonMap | list[JsonMap | str | int | float | bool | None] | str | int | float | bool | None


def workflow_payload(options: dict[str, str], flags: tuple[str, ...], session_id: str) -> JsonMap:
    payload: dict[str, JsonValue] = {
        "request_id": session_id,
        "user_goal": "Interactive ASA workflow harness",
        "evidence": _evidence_from_options(options),
    }
    for payload_key, option_keys in (
        ("actor_agent_id", ("actor_agent_id", "actor_agent", "caller_agent_id", "caller_agent")),
        ("caller_agent_id", ("caller_agent_id", "caller_agent", "actor_agent_id", "actor_agent")),
        ("owner_agent_id", ("owner_agent_id", "owner_agent")),
        ("target_agent_id", ("target_agent_id", "target_agent")),
        ("goal_id", ("goal_id", "goal")),
    ):
        value = option_value(options, *option_keys, default="")
        if value:
            payload[payload_key] = value
    if "artifact_root" in options:
        payload["artifact_root"] = options["artifact_root"]
    if _flag_enabled(options, flags, "validate_artifact_paths"):
        payload["validate_artifact_paths"] = True
    if "goals_path" in options:
        payload["goals_path"] = options["goals_path"]
    deep_interview = _deep_interview_from_options(options)
    if deep_interview:
        payload["deep_interview"] = deep_interview
    gate = _gate_from_options(options)
    if gate is not None:
        payload["gate"] = gate
    return payload


def option_value(options: dict[str, str], *keys: str, default: str = "orchestrator") -> str:
    for key in keys:
        value = options.get(key)
        if value:
            return value
    return default


def _evidence_from_options(options: dict[str, str]) -> JsonMap:
    evidence: dict[str, JsonValue] = {}
    raw = options.get("evidence_key", "")
    for item in raw.split(","):
        key = item.strip()
        if key:
            evidence[key] = "provided"
    for evidence_key, option_key in (
        ("surface_ref", "surface_ref"),
        ("screenshot_ref", "screenshot_ref"),
        ("oracle_verdict", "oracle_verdict"),
        ("capture_target", "capture_target"),
        ("research_question", "research_question"),
        ("source_journal", "source_journal"),
        ("insane_search_trace", "insane_search_trace"),
    ):
        if option_key in options:
            evidence[evidence_key] = _json_object_or_text(options[option_key])
    return evidence


def _gate_from_options(options: dict[str, str]) -> JsonMap | None:
    gate_id = options.get("gate_id", "").strip()
    if not gate_id:
        return None
    gate_kind = options.get("gate_kind", "enum").strip() or "enum"
    gate: dict[str, JsonValue] = {"gate_id": gate_id, "gate_kind": gate_kind}
    if gate_kind == "response_schema":
        gate["response_schema"] = _response_schema(options.get("response_schema", "{}"))
    else:
        gate["allowed_values"] = _csv_values(options.get("allowed_values", "approve,revise"))
    return gate


def _csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _response_schema(raw: str) -> JsonMap:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_object_or_text(raw: str) -> JsonMap | str:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return loaded if isinstance(loaded, dict) else raw


def _deep_interview_from_options(options: dict[str, str]) -> JsonMap:
    metadata: dict[str, JsonValue] = {}
    for payload_key, option_key in (
        ("round", "deep_round"),
        ("round_id", "deep_round_id"),
        ("component", "deep_component"),
        ("dimension", "deep_dimension"),
        ("ambiguity", "deep_ambiguity"),
        ("question_id", "deep_question_id"),
    ):
        value = options.get(option_key, "")
        if value:
            metadata[payload_key] = _numeric_value(value) if payload_key in {"round", "ambiguity"} else value
    if "deep_multi" in options:
        metadata["multi"] = _flag_enabled(options, (), "deep_multi")
    if "deep_options" in options:
        metadata["options"] = _csv_values(options["deep_options"])
    return metadata


def _flag_enabled(options: dict[str, str], flags: tuple[str, ...], key: str) -> bool:
    if key in flags:
        return True
    value = options.get(key, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _numeric_value(value: str) -> int | float | str:
    if value.isdecimal():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value
