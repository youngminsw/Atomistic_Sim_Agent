from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from sim_agent.schemas._parse import JsonMap

from .tui_paths import display_path
from .tui_state import SESSION_EVENTS_NAME, TuiState


GLOBAL_EVENTS_NAME = "global_session_events.jsonl"
AGENT_EVENTS_NAME = "events.jsonl"
SUBAGENT_RUN_NAME = "subagent_run.json"
SUBAGENT_CONTROLS_NAME = "subagent_controls.jsonl"


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    at: float
    source: str
    actor: str
    event_type: str
    summary: str
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class TimelineSummary:
    event_count: int
    latest_event_type: str
    latest_actor: str
    latest_source: str


def write_timeline(state: TuiState, output_stream: TextIO, *, limit: int = 30) -> None:
    events = timeline_events(state, limit=limit)
    output_stream.write("ASA Timeline\n")
    output_stream.write("timeline=true\n")
    output_stream.write(f"timeline_session_id={state.session_id}\n")
    output_stream.write(f"timeline_session_dir={display_path(state.session_dir)}\n")
    output_stream.write(f"timeline_event_count={len(events)}\n")
    for event in events:
        sequence = "-" if event.sequence is None else str(event.sequence)
        output_stream.write(
            "timeline_event="
            f"seq:{sequence} "
            f"source:{event.source} "
            f"actor:{event.actor} "
            f"type:{event.event_type} "
            f"summary:{_trim(event.summary)}\n"
        )


def timeline_summary(state: TuiState) -> TimelineSummary:
    events = timeline_events(state, limit=1)
    if not events:
        return TimelineSummary(0, "-", "-", "-")
    latest = events[-1]
    return TimelineSummary(
        event_count=len(timeline_events(state, limit=200)),
        latest_event_type=latest.event_type,
        latest_actor=latest.actor,
        latest_source=latest.source,
    )


def timeline_events(state: TuiState, *, limit: int = 30) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    events.extend(_session_events(state.session_dir / SESSION_EVENTS_NAME))
    events.extend(_global_events(state.session_dir / GLOBAL_EVENTS_NAME))
    events.extend(_agent_events(state.session_dir / "agent_sessions"))
    events.extend(_subagent_events(state.session_dir / "agent_sessions"))
    events.sort(key=lambda event: (event.at, event.sequence or 0, event.source, event.actor))
    if limit <= 0:
        return tuple(events)
    return tuple(events[-limit:])


def _session_events(path: Path) -> list[TimelineEvent]:
    rows: list[TimelineEvent] = []
    for payload in _read_jsonl(path):
        rows.append(
            TimelineEvent(
                at=_float(payload.get("at")),
                source="session",
                actor="tui",
                event_type=_str(payload.get("event_type"), "event"),
                summary=_str(payload.get("summary"), ""),
            )
        )
    return rows


def _global_events(path: Path) -> list[TimelineEvent]:
    rows: list[TimelineEvent] = []
    for payload in _read_jsonl(path):
        rows.append(
            TimelineEvent(
                at=_float(payload.get("at")),
                source="global",
                actor=_str(payload.get("actor"), "orchestrator"),
                event_type=_str(payload.get("event_type"), "event"),
                summary=_str(payload.get("summary"), ""),
                sequence=_int_or_none(payload.get("sequence")),
            )
        )
    return rows


def _agent_events(agent_root: Path) -> list[TimelineEvent]:
    rows: list[TimelineEvent] = []
    if not agent_root.is_dir():
        return rows
    for path in sorted(agent_root.glob(f"*/{AGENT_EVENTS_NAME}")):
        agent_id = path.parent.name
        for payload in _read_jsonl(path):
            rows.append(
                TimelineEvent(
                    at=_float(payload.get("at")),
                    source="agent",
                    actor=_str(payload.get("agent_id"), agent_id),
                    event_type=_str(payload.get("event_type"), "event"),
                    summary=_str(payload.get("summary"), ""),
                    sequence=_int_or_none(payload.get("sequence")),
                )
            )
    return rows


def _subagent_events(agent_root: Path) -> list[TimelineEvent]:
    rows: list[TimelineEvent] = []
    if not agent_root.is_dir():
        return rows
    for path in sorted(agent_root.glob(f"*/subagents/*/*/{SUBAGENT_RUN_NAME}")):
        payload = _read_json(path)
        if not payload:
            continue
        rows.append(
            TimelineEvent(
                at=_float(payload.get("at")),
                source="subagent",
                actor=f"{_str(payload.get('caller_agent'), '')}/{_str(payload.get('preset'), '')}",
                event_type="subagent_run",
                summary=f"{_str(payload.get('subagent_id'), path.parent.name)}:{_str(payload.get('status'), '')}",
            )
        )
    for path in sorted(agent_root.glob(f"*/subagents/*/*/{SUBAGENT_CONTROLS_NAME}")):
        for payload in _read_jsonl(path):
            rows.append(
                TimelineEvent(
                    at=_float(payload.get("at")),
                    source="subagent",
                    actor=f"{_str(payload.get('caller_agent'), '')}/{_str(payload.get('preset'), '')}",
                    event_type=f"subagent_{_str(payload.get('action'), 'control')}",
                    summary=_str(payload.get("content"), _str(payload.get("subagent_id"), path.parent.name)),
                )
            )
    return rows


def _read_jsonl(path: Path) -> list[JsonMap]:
    if not path.is_file():
        return []
    rows: list[JsonMap] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_json(path: Path) -> JsonMap:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _str(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _trim(value: str, limit: int = 160) -> str:
    cleaned = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
