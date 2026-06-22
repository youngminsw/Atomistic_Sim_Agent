from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.agent_runtime import (
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_global_session_event,
    open_global_session,
)
from sim_agent.cli.tui_state import append_event, initial_state


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_initial_state_resume_preserves_global_session_and_sequence(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    append_event(state, "user_turn", "first prompt")

    resumed = initial_state(tmp_path, resume="latest")
    append_event(resumed, "user_turn", "second prompt")

    assert resumed.session_id == state.session_id
    global_session = json.loads((tmp_path / "global_session.json").read_text(encoding="utf-8"))
    global_events = _jsonl(tmp_path / "global_session_events.jsonl")
    assert global_session["schema_version"] == "asa_global_session_v2"
    assert global_session["session_id"] == state.session_id
    assert global_session["last_sequence"] == 4
    assert [event["sequence"] for event in global_events] == [1, 2, 3, 4]
    assert [event["event_type"] for event in global_events] == [
        "session_created",
        "user_turn",
        "session_resumed",
        "user_turn",
    ]


def test_cli_resume_rehydrates_existing_global_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    first = _run_module_interactive(["/status", "/exit"], session_dir=session_dir)
    first_session = json.loads((session_dir / "asa_session.json").read_text(encoding="utf-8"))

    second = _run_module_interactive(["/status", "/exit"], session_dir=session_dir, extra_args=["--resume"])
    second_session = json.loads((session_dir / "asa_session.json").read_text(encoding="utf-8"))

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert second_session["session_id"] == first_session["session_id"]
    assert second_session["global_session_id"] == first_session["global_session_id"]
    global_events = _jsonl(session_dir / "global_session_events.jsonl")
    assert any(event["event_type"] == "session_resumed" for event in global_events)
    assert [event["sequence"] for event in global_events] == list(range(1, len(global_events) + 1))


def test_global_session_store_resumes_specific_session_id_from_index(tmp_path: Path) -> None:
    model = _model()
    created = open_global_session(GlobalSessionOpenRequest(requested_dir=None, default_root=tmp_path, model=model))
    append_global_session_event(created.record.session_dir, "user_turn", "first prompt")

    resumed = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=None,
            default_root=tmp_path,
            model=model,
            resume=created.record.session_id,
        )
    )
    append_global_session_event(resumed.record.session_dir, "user_turn", "second prompt")

    assert resumed.record.session_id == created.record.session_id
    global_session = json.loads((created.record.session_dir / "global_session.json").read_text(encoding="utf-8"))
    global_events = _jsonl(created.record.session_dir / "global_session_events.jsonl")
    assert global_session["last_sequence"] == 2
    assert [event["sequence"] for event in global_events] == [1, 2]


def _run_module_interactive(
    lines: list[str],
    *,
    session_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_STARTUP_WIZARD"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "sim_agent", *(extra_args or [])],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _model() -> GlobalSessionModel:
    return GlobalSessionModel(
        provider="openai-codex",
        name="gpt-5-codex",
        reasoning_effort="high",
        base_url="https://model-gateway.local/v1",
        auth_mode="gateway",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )
