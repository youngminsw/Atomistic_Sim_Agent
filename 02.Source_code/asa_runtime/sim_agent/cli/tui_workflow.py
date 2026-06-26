from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke, workflow_harness_catalog
from sim_agent.agents_sdk_runtime.workflow_gate_protocol import safe_id

from .tui_parse import parse_options
from .tui_state import TuiState, append_event


WORKFLOW_ALIASES: tuple[str, ...] = tuple(workflow.workflow_id for workflow in workflow_harness_catalog())


def handle_workflow(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    workflow_id = _workflow_id(parsed.remainder)
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "workflows")))
    payload = _workflow_payload(parsed.options, parsed.flags, state.session_id)
    result = run_workflow_harness_smoke(
        workflow_id,
        payload,
        output_dir,
    )
    append_event(state, "workflow_harness", f"{result.workflow_id}:{result.status}")
    output_stream.write(
        f"Workflow {result.workflow_id}: {result.status} "
        f"(state {result.current_state}, gate {result.gate_status})\n"
    )
    output_stream.write("workflow_harness_ready=true\n")
    output_stream.write(f"workflow={result.workflow_id}\n")
    output_stream.write(f"workflow_status={result.status}\n")
    output_stream.write(f"current_state={result.current_state}\n")
    output_stream.write(f"workflow_loop_state={result.current_state}\n")
    output_stream.write(f"verification_gate={result.verification_gate}\n")
    output_stream.write(f"workflow_gate_status={result.gate_status}\n")
    output_stream.write(f"workflow_actor_agent_id={result.actor_agent_id}\n")
    output_stream.write(f"workflow_owner_agent_id={result.owner_agent_id}\n")
    output_stream.write(f"workflow_target_agent_id={result.target_agent_id}\n")
    if result.goal_id:
        output_stream.write(f"workflow_goal_id={result.goal_id}\n")
    if result.gate is not None:
        output_stream.write(f"workflow_gate_id={result.gate.get('gate_id', '')}\n")
        output_stream.write(f"workflow_gate_kind={result.gate.get('gate_kind', '')}\n")
        output_stream.write(f"workflow_gate_schema_hash={result.gate.get('schema_hash', '')}\n")
        output_stream.write(f"workflow_gate_ledger_ref={result.gate.get('ledger_ref', '')}\n")
        metadata = result.gate.get("deep_interview")
        if isinstance(metadata, dict):
            output_stream.write(f"workflow_deep_interview_round={metadata.get('round', '')}\n")
            output_stream.write(f"workflow_deep_interview_round_id={metadata.get('round_id', '')}\n")
            output_stream.write(f"workflow_deep_interview_component={metadata.get('component', '')}\n")
            output_stream.write(f"workflow_deep_interview_dimension={metadata.get('dimension', '')}\n")
            output_stream.write(f"workflow_deep_interview_ambiguity={metadata.get('ambiguity', '')}\n")
    if result.evidence_keys:
        output_stream.write(f"workflow_evidence_keys={','.join(result.evidence_keys)}\n")
    if result.artifact_refs:
        output_stream.write(f"workflow_artifact_refs={','.join(result.artifact_refs)}\n")
    if result.missing_evidence:
        output_stream.write(f"workflow_missing_evidence={','.join(result.missing_evidence)}\n")
    output_stream.write(f"workflow_ledger_path={output_dir / result.ledger_ref}\n")
    for blocker in result.blockers:
        output_stream.write(f"workflow_blocker={blocker}\n")
    return state


def handle_workflow_response(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    if len(parsed.remainder) < 2:
        append_event(state, "workflow_gate_response_blocked", "missing_gate_or_value")
        output_stream.write("workflow_response=false\n")
        output_stream.write("workflow_response_error=missing_gate_or_value\n")
        return state
    workflow_id = _option_value(parsed.options, "workflow_id", "workflow", default="deep-interview")
    gate_id, value_text = parsed.remainder[0], parsed.remainder[1]
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "workflows")))
    responder_agent_id = _option_value(
        parsed.options,
        "responder_agent_id",
        "responder_agent",
        default="orchestrator",
    )
    result = respond_workflow_gate(
        output_dir,
        {
            "workflow_id": workflow_id,
            "gate_id": gate_id,
            "responder_agent_id": responder_agent_id,
            "value": _response_value(value_text, output_dir, workflow_id, gate_id),
        },
    )
    append_event(state, "workflow_gate_response", f"{result.workflow_id}:{result.gate_id}:{result.status}")
    output_stream.write("workflow_response=true\n")
    output_stream.write(f"workflow_response_status={result.status}\n")
    output_stream.write(f"workflow={result.workflow_id}\n")
    output_stream.write(f"workflow_gate_id={result.gate_id}\n")
    output_stream.write(f"workflow_gate_status={result.status}\n")
    output_stream.write(f"workflow_owner_agent_id={result.owner_agent_id}\n")
    output_stream.write(f"workflow_target_agent_id={result.target_agent_id}\n")
    output_stream.write(f"workflow_answered_at={result.answered_at}\n")
    output_stream.write(f"workflow_ledger_path={output_dir / result.ledger_ref}\n")
    for blocker in result.blockers:
        output_stream.write(f"workflow_blocker={blocker}\n")
    return state


def write_workflow_catalog(output_stream: TextIO) -> None:
    output_stream.write("Workflow Catalog\n")
    output_stream.write("Workflow             Current State                 Verification Gate\n")
    output_stream.write("workflow_catalog=true\n")
    for workflow in workflow_harness_catalog():
        output_stream.write(
            f"{workflow.workflow_id:<20} {workflow.current_state:<29} {workflow.verification_gate}\n"
        )
        output_stream.write(
            f"workflow={workflow.workflow_id} current_state={workflow.current_state} "
            f"verification_gate={workflow.verification_gate}\n"
        )


def _workflow_id(remainder: tuple[str, ...]) -> str:
    if remainder:
        return remainder[0]
    return "deep-interview"


def _workflow_payload(options: dict[str, str], flags: tuple[str, ...], session_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
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
        value = _option_value(options, *option_keys, default="")
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


def _evidence_from_options(options: dict[str, str]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    raw = options.get("evidence_key", "")
    for item in raw.split(","):
        key = item.strip()
        if key:
            evidence[key] = "provided"
    return evidence


def _gate_from_options(options: dict[str, str]) -> dict[str, object] | None:
    gate_id = options.get("gate_id", "").strip()
    if not gate_id:
        return None
    gate_kind = options.get("gate_kind", "enum").strip() or "enum"
    gate: dict[str, object] = {"gate_id": gate_id, "gate_kind": gate_kind}
    if gate_kind == "response_schema":
        gate["response_schema"] = _response_schema(options.get("response_schema", "{}"))
    else:
        gate["allowed_values"] = _csv_values(options.get("allowed_values", "approve,revise"))
    return gate


def _csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _response_schema(raw: str) -> dict[str, object]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _deep_interview_from_options(options: dict[str, str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
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


def _response_value(raw: str, output_dir: Path, workflow_id: str, gate_id: str) -> object:
    stripped = raw.strip()
    if _stored_gate_kind(output_dir, workflow_id, gate_id) == "response_schema":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    if not stripped.startswith(("{", "[", '"')):
        return raw
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def _stored_gate_kind(output_dir: Path, workflow_id: str, gate_id: str) -> str:
    gate_path = output_dir / safe_id(workflow_id) / "gates" / f"{safe_id(gate_id)}.json"
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("gate_kind"), str):
        return payload["gate_kind"]
    return ""


def _flag_enabled(options: dict[str, str], flags: tuple[str, ...], key: str) -> bool:
    if key in flags:
        return True
    value = options.get(key, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _option_value(options: dict[str, str], *keys: str, default: str = "orchestrator") -> str:
    for key in keys:
        value = options.get(key)
        if value:
            return value
    return default


def _numeric_value(value: str) -> int | float | str:
    if value.isdecimal():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value
