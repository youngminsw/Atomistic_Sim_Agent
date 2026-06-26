from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from sim_agent.cli.tui_chat import handle_chat_message
from sim_agent.cli.tui_agent_activity import build_agent_activity_summary
from sim_agent.cli.tui_team import handle_agents
from sim_agent.cli.tui_state import ModelSettings, TuiState, append_event, initial_state


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_agent_activity_summary_marks_latest_summoned_specialist(tmp_path: Path) -> None:
    state = _state(tmp_path)
    append_event(state, "agent_direct_route", "md_agent")

    summary = build_agent_activity_summary(state, heartbeat_s=3600)
    rows = {row.agent_id: row for row in summary.rows}

    assert summary.mode == "direct_session"
    assert summary.active_agent == "md_agent"
    assert rows["orchestrator"].status == "ready"
    assert rows["md_agent"].status == "direct_session"
    assert rows["md_agent"].activity == "persistent agent session accepted message"
    assert rows["qa_agent"].status == "ready"


def test_agent_activity_summary_ignores_corrupt_event_lines(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.session_dir.mkdir(parents=True, exist_ok=True)
    (state.session_dir / "asa_session_events.jsonl").write_text(
        "{broken-json\n"
        '{"event_type": "agent_direct_route", "summary": "qa_agent"}\n',
        encoding="utf-8",
    )

    summary = build_agent_activity_summary(state)

    assert summary.mode == "direct_session"
    assert summary.active_agent == "qa_agent"


def test_tui_agents_and_hud_show_direct_agent_session(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@md_agent check receipt mode\n/agents\n/hud\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_direct_route=md_agent" in result.stdout
    assert "agent_session_mode=persistent" in result.stdout
    assert "agent_activity_mode=direct_session" in result.stdout
    assert "active_agent=md_agent" in result.stdout
    assert "agent_activity=md_agent direct_session" in result.stdout
    assert "agent_activity_label=md_agent · direct-session · persistent handle" in result.stdout
    assert "routed_via=orchestrator" not in result.stdout
    assert "receipt_only" not in result.stdout


def test_agents_command_separates_persistent_agents_from_bounded_presets(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/agents\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    persistent_index = result.stdout.index("Persistent Domain Agents")
    bounded_index = result.stdout.index("Global Bounded Subagent Presets")
    assert persistent_index < bounded_index
    assert "orchestrator" in result.stdout
    assert "md_agent" in result.stdout
    assert "qa_agent" in result.stdout
    assert "planner" in result.stdout
    assert "architect" in result.stdout
    assert "critic" in result.stdout
    assert "executor" in result.stdout
    assert "persistent session" in result.stdout
    assert "clean-room bounded" in result.stdout


def test_agents_tty_output_hides_semantic_debug_lines(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    output = _TtyStringIO()

    handle_agents(state, output)

    rendered = output.getvalue()
    assert "Persistent Domain Agents" in rendered
    assert "Global Bounded Subagent Presets" in rendered
    assert "agent_roster=" not in rendered
    assert "agent_session=" not in rendered
    assert "agent_registry_path=" not in rendered
    old_threshold_line = "_".join(("auto", "compaction", "new", "message", "threshold="))
    assert old_threshold_line not in rendered


def test_direct_agent_mention_appends_target_agent_transcript(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@qa_agent audit the current MD evidence\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    messages = _jsonl(session_dir / "agent_sessions" / "qa_agent" / "messages.jsonl")
    events = _jsonl(session_dir / "agent_sessions" / "qa_agent" / "events.jsonl")
    assert any(
        message["role"] == "user" and message["content"] == "audit the current MD evidence"
        for message in messages
    )
    assert messages[-1]["role"] == "assistant"
    assert "agent loop blocked: endpoint_unreachable" in str(messages[-1]["content"])
    assert events[-1]["event_type"] == "agent_message_appended"


def test_direct_agent_mention_runs_target_agent_loop(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@md_agent plan a tiny test\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    messages = _jsonl(session_dir / "agent_sessions" / "md_agent" / "messages.jsonl")
    events = _jsonl(session_dir / "agent_sessions" / "md_agent" / "events.jsonl")
    event_types = {event["event_type"] for event in events}
    assert "agent_loop_completed" in event_types
    assert any(message["role"] == "assistant" and "agent loop" in str(message["content"]) for message in messages)
    assert "agent_loop_status=blocked" in result.stdout
    assert "endpoint_unreachable" in result.stdout
    assert "agent_loop_tools=" in result.stdout
    assert "ASA Activity Rail" in result.stdout
    assert "Progress" in result.stdout
    assert "Tool Call" in result.stdout
    assert "Output" in result.stdout
    assert "Runtime Events" in result.stdout
    assert "tool_call=none" in result.stdout
    assert "event=model_start" in result.stdout
    assert "event=blocker" in result.stdout
    assert "ASA Chat Deck" in result.stdout


def test_direct_agent_tty_output_hides_semantic_debug_lines(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output = _TtyStringIO()

    handle_chat_message(
        ["@md_agent", "show", "TTY", "response"],
        state,
        output,
        _unused_run_handler,
    )

    rendered = output.getvalue()
    assert "ASA Activity Rail" in rendered
    assert "Tool Call" in rendered
    assert "Runtime Events" in rendered
    assert "ASA Chat Deck" in rendered
    assert "assistant@md_agent" in rendered
    assert "agent_direct_route=" not in rendered
    assert "agent_loop_status=" not in rendered
    assert "tool_call=" not in rendered
    assert "event=model_start" not in rendered
    assert "chat_window=true" not in rendered
    assert "chat_message_count=" not in rendered


def test_direct_agent_mention_continues_same_agent_session(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@md_agent first turn\n@md_agent second turn\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    messages = _jsonl(session_dir / "agent_sessions" / "md_agent" / "messages.jsonl")
    events = _jsonl(session_dir / "agent_sessions" / "md_agent" / "events.jsonl")
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert [message["content"] for message in messages[::2]] == ["first turn", "second turn"]
    assert len({message["agent_session_id"] for message in messages}) == 1
    assert sum(1 for event in events if event["event_type"] == "agent_loop_completed") == 2


def test_unknown_agent_mention_blocks_without_orchestrator_run(tmp_path: Path) -> None:
    state = _state(tmp_path)
    output = StringIO()
    legacy_agent_id = "ml_" + "mdn_agent"

    handle_chat_message(
        [f"@{legacy_agent_id}", "legacy", "alias"],
        state,
        output,
        _unused_run_handler,
    )

    rendered = output.getvalue()
    assert f"agent_direct_route_blocked={legacy_agent_id}" in rendered
    assert "agent_direct_route_blocker=unknown_agent" in rendered
    assert not (tmp_path / "chat-runs").exists()


def _state(tmp_path: Path) -> TuiState:
    return TuiState(session_id="asa-test", session_dir=tmp_path, model=ModelSettings())


def _unused_run_handler(args: list[str], state: TuiState, output_stream: StringIO) -> TuiState:
    raise AssertionError("direct agent mention must not call orchestrator run handler")


class _TtyStringIO(StringIO):
    @property
    def encoding(self) -> str:
        return "utf-8"

    def isatty(self) -> bool:
        return True


def _env(tmp_path: Path) -> dict[str, str]:
    runtime_config = tmp_path / "runtime-config.json"
    runtime_config.write_text(
        json.dumps(
            {
                "model_endpoint": {
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "reasoning_effort": "high",
                    "base_url": "http://127.0.0.1:1/v1",
                    "auth_mode": "oauth",
                    "api_key_env": "ASA_OPENAI_CODEX_TOKEN",
                }
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(runtime_config)
    env["ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE"] = str(tmp_path / "credentials.json")
    env.pop("MODEL_GATEWAY_TOKEN", None)
    return env


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
