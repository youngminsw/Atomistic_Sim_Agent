from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .workflow_gate_protocol import WorkflowGate, WorkflowGateKind, gate_ledger_ref, now, read_gate, safe_id, workflow_gate_schema_hash


AMBIGUITY_HANDOFF_THRESHOLD: Final = 0.20
DEFAULT_MAX_ROUNDS: Final = 5
QUESTION_RE: Final = re.compile(
    r"Round\s+(?P<round>\d+).*?"
    r"round_id=(?P<round_id>[A-Za-z0-9_.-]+).*?"
    r"component=(?P<component>[A-Za-z0-9_.-]+).*?"
    r"dimension=(?P<dimension>[A-Za-z0-9_.-]+).*?"
    r"ambiguity=(?P<ambiguity>0(?:\.\d+)?|1(?:\.0+)?) .*?"
    r"question_id=(?P<question_id>[A-Za-z0-9_.-]+).*?"
    r"multi=(?P<multi>true|false).*?"
    r"options=(?P<options>[A-Za-z0-9_.| -]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DeepInterviewDecodedAnswer:
    selected_options: tuple[str, ...]
    custom_input: str
    handoff_ready: bool


def deep_interview_corrupt_state(workflow_dir: Path) -> bool:
    state_dir = workflow_dir / "state"
    if not state_dir.is_dir():
        return False
    for path in state_dir.glob("*.json"):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return True
        if not isinstance(loaded, dict):
            return True
    return False


def deep_interview_pending_gate(workflow_dir: Path) -> WorkflowGate | None:
    gates_dir = workflow_dir / "gates"
    if not gates_dir.is_dir():
        return None
    for path in sorted(gates_dir.glob("*.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(loaded, dict) and loaded.get("status") == "awaiting_response":
            return read_gate(path)
    return None


def deep_interview_gate(
    workflow_id: str,
    goal_id: str,
    owner_agent_id: str,
    target_agent_id: str,
    payload: JsonMap,
) -> WorkflowGate:
    metadata = deep_interview_metadata(payload)
    gate_id = f"question-{metadata['question_id']}"
    gate = WorkflowGate(
        workflow_id,
        goal_id,
        safe_id(gate_id),
        WorkflowGateKind.RESPONSE_SCHEMA,
        owner_agent_id,
        target_agent_id,
        "awaiting_response",
        now(),
        gate_ledger_ref(workflow_id, safe_id(gate_id)),
        (),
        (),
        _answer_schema(metadata),
        deep_interview=metadata,
    )
    return WorkflowGate(
        gate.workflow_id,
        gate.goal_id,
        gate.gate_id,
        gate.gate_kind,
        gate.owner_agent_id,
        gate.target_agent_id,
        gate.status,
        gate.created_at,
        gate.ledger_ref,
        gate.blockers,
        gate.allowed_values,
        gate.response_schema,
        gate.answered_at,
        workflow_gate_schema_hash(gate),
        gate.deep_interview,
    )


def deep_interview_metadata(payload: JsonMap) -> dict[str, object]:
    explicit = payload.get("deep_interview")
    if isinstance(explicit, dict):
        return _metadata_from_mapping(explicit, payload)
    question = payload.get("question")
    if isinstance(question, str):
        matched = QUESTION_RE.search(question)
        if matched is not None:
            return _metadata_from_regex(matched, payload)
    return _metadata_from_mapping({}, payload)


def deep_interview_response_blocker(gate: WorkflowGate, value: object) -> str:
    metadata = gate.deep_interview
    if metadata is None:
        return ""
    decoded = decode_deep_interview_answer(metadata, value)
    return "" if decoded is not None else "workflow_gate_response_schema_mismatch"


def decode_deep_interview_answer(metadata: JsonMap, value: object) -> DeepInterviewDecodedAnswer | None:
    if not isinstance(value, dict):
        return None
    allowed_keys = {"selected", "other", "custom", "handoff_ready"}
    if any(not isinstance(key, str) or key not in allowed_keys for key in value):
        return None
    selected_value = value.get("selected")
    if not isinstance(selected_value, list) or any(not isinstance(item, str) for item in selected_value):
        return None
    selected = tuple(selected_value)
    other = value.get("other") is True
    custom = value.get("custom")
    custom_text = custom.strip() if isinstance(custom, str) else ""
    options = tuple(item for item in metadata.get("options", ()) if isinstance(item, str))
    if any(item not in options for item in selected):
        return None
    if other:
        if selected or not custom_text:
            return None
        return DeepInterviewDecodedAnswer((), custom_text, value.get("handoff_ready") is True)
    if custom_text:
        return None
    if not selected:
        return None
    if metadata.get("multi") is not True and len(selected) != 1:
        return None
    return DeepInterviewDecodedAnswer(selected, "", value.get("handoff_ready") is True)


def record_deep_interview_response(output_dir: Path, gate: WorkflowGate, value: object) -> None:
    metadata = gate.deep_interview
    if metadata is None:
        return
    decoded = decode_deep_interview_answer(metadata, value)
    if decoded is None:
        return
    workflow_dir = output_dir / "deep-interview"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    blocker = _handoff_blocker(metadata, decoded)
    row: dict[str, object] = {
        "workflow_id": "deep-interview",
        "round": metadata["round"],
        "round_id": metadata["round_id"],
        "component": metadata["component"],
        "dimension": metadata["dimension"],
        "ambiguity": metadata["ambiguity"],
        "question_id": metadata["question_id"],
        "selected_options": list(decoded.selected_options),
        "custom_input": decoded.custom_input,
        "answered_at": now(),
    }
    if blocker:
        row["blocker"] = blocker
    with (workflow_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(workflow_dir / "state" / f"{safe_id(str(metadata['round_id']))}.json", row | {"status": "accepted"})
    if _handoff_ready(metadata, decoded):
        _write_handoff(workflow_dir / "handoff.md", row, blocker)


def deep_interview_handoff_refs(workflow_dir: Path) -> tuple[str, ...]:
    if (workflow_dir / "handoff.md").is_file():
        return ("deep-interview/transcript.jsonl", "deep-interview/handoff.md")
    return ()


def _metadata_from_mapping(value: JsonMap, payload: JsonMap) -> dict[str, object]:
    round_number = _int_value(value.get("round"), 1)
    question_id = _text_value(value.get("question_id"), f"q{round_number}")
    return {
        "round": round_number,
        "round_id": _text_value(value.get("round_id"), f"round-{round_number}"),
        "component": _text_value(value.get("component"), "requirements"),
        "dimension": _text_value(value.get("dimension"), "scope"),
        "ambiguity": _float_value(value.get("ambiguity"), 1.0),
        "question_id": question_id,
        "multi": value.get("multi") is True,
        "options": _options(value.get("options")),
        "max_rounds": _int_value(payload.get("max_rounds"), DEFAULT_MAX_ROUNDS),
    }


def _metadata_from_regex(matched: re.Match[str], payload: JsonMap) -> dict[str, object]:
    return _metadata_from_mapping(
        {
            "round": matched.group("round"),
            "round_id": matched.group("round_id"),
            "component": matched.group("component"),
            "dimension": matched.group("dimension"),
            "ambiguity": matched.group("ambiguity"),
            "question_id": matched.group("question_id"),
            "multi": matched.group("multi").lower() == "true",
            "options": matched.group("options").split("|"),
        },
        payload,
    )


def _answer_schema(metadata: JsonMap) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected"],
        "properties": {
            "selected": {"type": "array", "items": {"type": "string", "enum": list(metadata["options"])}},
            "other": {"type": "boolean"},
            "custom": {"type": "string"},
            "handoff_ready": {"type": "boolean"},
        },
    }


def _handoff_ready(metadata: JsonMap, decoded: DeepInterviewDecodedAnswer) -> bool:
    return bool(decoded.handoff_ready or _float_value(metadata.get("ambiguity"), 1.0) <= AMBIGUITY_HANDOFF_THRESHOLD or _handoff_blocker(metadata, decoded))


def _handoff_blocker(metadata: JsonMap, decoded: DeepInterviewDecodedAnswer) -> str:
    if decoded.handoff_ready or _float_value(metadata.get("ambiguity"), 1.0) <= AMBIGUITY_HANDOFF_THRESHOLD:
        return ""
    if _int_value(metadata.get("round"), 1) >= _int_value(metadata.get("max_rounds"), DEFAULT_MAX_ROUNDS):
        return "deep_interview_max_rounds_reached"
    return ""


def _write_handoff(path: Path, row: JsonMap, blocker: str) -> None:
    lines = ["# Deep Interview Handoff", "", f"- round: {row['round']}", f"- round_id: {row['round_id']}", f"- ambiguity: {row['ambiguity']}"]
    if row.get("selected_options"):
        lines.append(f"- selected_options: {', '.join(str(item) for item in row['selected_options'])}")
    if row.get("custom_input"):
        lines.append(f"- custom_input: {row['custom_input']}")
    if blocker:
        lines.append(f"- blocker: {blocker}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text_value(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _int_value(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return default


def _float_value(value: object, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _options(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str) and item]
    return ["Clarify scope", "Clarify constraints"]
