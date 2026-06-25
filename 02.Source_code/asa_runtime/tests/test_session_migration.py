from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sim_agent.agent_runtime import (
    CompactionRequest,
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_agent_message,
    append_global_session_event,
    compact_agent_session,
    open_global_session,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.global_session_store import SCHEMA_BACKUP_SUFFIX, paths_for


def test_new_session_writes_v2_markers_and_v3_compaction_cursor(tmp_path: Path) -> None:
    created = open_global_session(GlobalSessionOpenRequest(requested_dir=tmp_path, default_root=tmp_path, model=_model()))
    append_global_session_event(created.record.session_dir, "user_turn", "hello")
    append_agent_message(created.record.session_dir, "md_agent", "user", "etch silicon")
    compact_agent_session(
        created.record.session_dir,
        CompactionRequest(agent_id="md_agent", compact_id="compact-md-v2", summary="md transcript"),
    )

    global_session = _json(created.record.paths.global_session)
    global_events = _jsonl(created.record.paths.global_events)
    registry = _json(created.record.session_dir / "agent_registry.json")
    agent_session = _json(created.record.session_dir / "agent_sessions" / "md_agent" / "session.json")
    messages = _jsonl(created.record.session_dir / "agent_sessions" / "md_agent" / "messages.jsonl")
    agent_events = _jsonl(created.record.session_dir / "agent_sessions" / "md_agent" / "events.jsonl")
    compact_summary = _json(created.record.session_dir / "agent_sessions" / "md_agent" / "compact_summary.json")
    compact_ledger = _jsonl(created.record.session_dir / "agent_sessions" / "md_agent" / "compactions.jsonl")

    assert global_session["schema_version"] == "asa_global_session_v2"
    assert global_events[-1]["schema_version"] == "asa_global_session_event_v2"
    assert registry["schema_version"] == "asa_agent_registry_v2"
    assert agent_session["schema_version"] == "asa_agent_session_v2"
    assert messages[-1]["schema_version"] == "asa_agent_chat_message_v2"
    assert agent_events[-1]["schema_version"] == "asa_agent_session_event_v2"
    assert compact_summary["schema_version"] == "asa_agent_compact_summary_v3"
    assert compact_summary["first_kept_message_sequence"] == 1
    assert compact_ledger[-1]["schema_version"] == "asa_agent_compact_summary_v3"


def test_resume_v1_global_session_from_latest_id_and_path_backs_up_before_v2_write(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "asa-old"
    session_dir.mkdir(parents=True)
    paths = paths_for(session_dir)
    old_payload = {
        "schema_version": "asa_global_session_v1",
        "session_id": "asa-old",
        "session_dir": str(session_dir),
        "created_at": 1.0,
        "updated_at": 1.0,
        "last_sequence": 0,
        "model": asdict(_model()),
        "agent_ids": ["orchestrator"],
        "paths": {
            "global_session": str(paths.global_session),
            "global_events": str(paths.global_events),
            "legacy_session": str(paths.legacy_session),
            "agent_sessions": str(paths.agent_sessions),
            "message_bus": str(paths.message_bus),
        },
        "source": "created",
    }
    old_bytes = _write_json(paths.global_session, old_payload)
    _append_jsonl(tmp_path / "global_session_index.jsonl", {"at": 1.0, "session_id": "asa-old", "session_dir": str(session_dir)})

    latest = open_global_session(GlobalSessionOpenRequest(requested_dir=None, default_root=tmp_path, model=_model(), resume="latest"))
    by_id = open_global_session(GlobalSessionOpenRequest(requested_dir=None, default_root=tmp_path, model=_model(), resume="asa-old"))
    by_path = open_global_session(GlobalSessionOpenRequest(requested_dir=None, default_root=tmp_path, model=_model(), resume=str(session_dir)))

    backup_path = paths.global_session.with_name(f"global_session.json{SCHEMA_BACKUP_SUFFIX}")
    migrated = _json(paths.global_session)
    assert latest.record.session_id == "asa-old"
    assert by_id.record.session_id == "asa-old"
    assert by_path.record.session_id == "asa-old"
    assert backup_path.read_bytes() == old_bytes
    assert migrated["schema_version"] == "asa_global_session_v2"


def test_resume_markerless_agent_session_backs_up_and_appends_v2_event_without_rewriting_old_events(tmp_path: Path) -> None:
    created = open_global_session(GlobalSessionOpenRequest(requested_dir=tmp_path, default_root=tmp_path, model=_model()))
    agent_dir = created.record.session_dir / "agent_sessions" / "md_agent"
    session_path = agent_dir / "session.json"
    old_session = _json(session_path)
    old_session.pop("schema_version")
    old_session_bytes = _write_json(session_path, old_session)
    events_path = agent_dir / "events.jsonl"
    old_event_line = json.dumps({"at": 1.0, "sequence": 1, "agent_id": "md_agent", "agent_session_id": "asa-old:md_agent", "event_type": "old", "summary": "old"}, sort_keys=True)
    events_path.write_text(f"{old_event_line}\n", encoding="utf-8")

    open_global_session(GlobalSessionOpenRequest(requested_dir=tmp_path, default_root=tmp_path, model=_model(), resume="latest"))

    event_lines = events_path.read_text(encoding="utf-8").splitlines()
    backup_path = session_path.with_name(f"session.json{SCHEMA_BACKUP_SUFFIX}")
    assert backup_path.read_bytes() == old_session_bytes
    assert _json(session_path)["schema_version"] == "asa_agent_session_v2"
    assert event_lines[0] == old_event_line
    assert json.loads(event_lines[-1])["schema_version"] == "asa_agent_session_event_v2"


def test_chat_transcript_reader_keeps_old_lines_and_appends_v2_message(tmp_path: Path) -> None:
    created = open_global_session(GlobalSessionOpenRequest(requested_dir=tmp_path, default_root=tmp_path, model=_model()))
    messages_path = created.record.session_dir / "agent_sessions" / "qa_agent" / "messages.jsonl"
    old_message_line = json.dumps({"at": 1.0, "sequence": 1, "agent_id": "qa_agent", "agent_session_id": "old", "role": "user", "content": "legacy"}, sort_keys=True)
    messages_path.write_text(f"{old_message_line}\n", encoding="utf-8")

    append_agent_message(created.record.session_dir, "qa_agent", "assistant", "new answer")

    lines = messages_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == old_message_line
    assert json.loads(lines[-1])["schema_version"] == "asa_agent_chat_message_v2"
    assert json.loads(lines[-1])["sequence"] == 2


def test_replay_v1_compaction_summary_backs_up_before_v3_rewrite(tmp_path: Path) -> None:
    created = open_global_session(GlobalSessionOpenRequest(requested_dir=tmp_path, default_root=tmp_path, model=_model()))
    append_agent_message(created.record.session_dir, "feature_scale_agent", "user", "legacy compact seed")
    agent_dir = created.record.session_dir / "agent_sessions" / "feature_scale_agent"
    summary_path = agent_dir / "compact_summary.json"
    old_summary = {
        "schema_version": "asa_agent_compact_summary_v1",
        "compact_id": "legacy-compact",
        "agent_id": "feature_scale_agent",
        "agent_session_id": f"{created.record.session_id}:feature_scale_agent",
        "compact_mode": "manual",
        "summary": "old summary",
        "message_count": 1,
        "event_count": 1,
        "last_message_sequence": 1,
        "last_event_sequence": 1,
        "created_at": 1.0,
    }
    old_bytes = _write_json(summary_path, old_summary)

    replayed = replay_agent_compaction(created.record.session_dir, "feature_scale_agent")

    backup_path = summary_path.with_name(f"compact_summary.json{SCHEMA_BACKUP_SUFFIX}")
    migrated = _json(summary_path)
    assert replayed.status == "succeeded"
    assert backup_path.read_bytes() == old_bytes
    assert migrated["schema_version"] == "asa_agent_compact_summary_v3"
    assert migrated["first_kept_message_sequence"] == 1
    assert migrated["manual_replay_status"] == "passed"


def _model() -> GlobalSessionModel:
    return GlobalSessionModel(
        provider="openai-codex",
        name="gpt-5-codex",
        reasoning_effort="high",
        base_url="https://model-gateway.local/v1",
        auth_mode="gateway",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    return content.encode()


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
