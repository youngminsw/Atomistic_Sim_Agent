from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.cli.tui_state import initial_state
from sim_agent.cli.tui_timeline import timeline_events
from sim_agent.runtime_config import default_runtime_config, save_runtime_config
from sim_agent.runtime_config_types import ModelEndpointRuntimeConfig


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_tui_timeline_renders_global_session_agent_and_hud_events(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    result = _run_module_interactive(
        ["@md_agent plan a tiny run", "/timeline --limit 80", "/hud", "/exit"],
        session_dir=session_dir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "timeline=true" in result.stdout
    assert "timeline_event_count=" in result.stdout
    assert "source:global" in result.stdout
    assert "source:session" in result.stdout
    assert "source:agent actor:md_agent" in result.stdout
    assert "type:agent_loop_completed" in result.stdout
    assert "timeline rail" in result.stdout
    assert "timeline_latest=" in result.stdout


def test_tui_resume_command_reopens_current_global_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    first = _run_module_interactive(["/status", "/exit"], session_dir=session_dir)
    created = json.loads((session_dir / "global_session.json").read_text(encoding="utf-8"))

    second = _run_module_interactive(["/resume latest", "/status", "/exit"], session_dir=session_dir, extra_args=["--resume"])
    resumed = json.loads((session_dir / "global_session.json").read_text(encoding="utf-8"))
    global_events = _jsonl(session_dir / "global_session_events.jsonl")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Session Resumed" in second.stdout
    assert "resume=true" in second.stdout
    assert resumed["session_id"] == created["session_id"]
    assert any(event["event_type"] == "session_resumed" for event in global_events)
    assert [event["sequence"] for event in global_events] == list(range(1, len(global_events) + 1))


def test_timeline_collects_subagent_run_and_control_events(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    registry = default_tool_registry()
    execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_task",
            arguments={
                "preset": "critic",
                "task_id": "timeline-review",
                "task": "Review timeline evidence.",
            },
            run_id="subagent-run",
            session_id=state.session_id,
            caller_agent_id="qa_agent",
        ),
        registry,
        state.session_dir,
    )
    running_dir = state.session_dir / "agent_sessions" / "qa_agent" / "subagents" / "critic" / "timeline-live"
    running_dir.mkdir(parents=True)
    (running_dir / "subagent_running.lock").write_text(
        json.dumps({"caller_agent": "qa_agent", "preset": "critic", "subagent_id": "timeline-live"}) + "\n",
        encoding="utf-8",
    )
    steered = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="subagent_control",
            arguments={
                "action": "steer",
                "preset": "critic",
                "subagent_id": "timeline-live",
                "content": "Inspect runtime spine evidence.",
            },
            run_id="steer-run",
            session_id=state.session_id,
            caller_agent_id="qa_agent",
        ),
        registry,
        state.session_dir,
    )

    events = timeline_events(state, limit=50)

    assert steered.status == "blocked"
    assert steered.blocker == "subagent_control_unsupported"
    assert any(event.source == "subagent" and event.event_type == "subagent_run" for event in events)
    assert not any(event.source == "subagent" and event.event_type == "subagent_steer" for event in events)


def _run_module_interactive(
    lines: list[str],
    *,
    session_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    config_path = session_dir / "runtime-config.json"
    _write_static_runtime_config(config_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
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


def _write_static_runtime_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = default_runtime_config()
    save_runtime_config(
        replace(
            config,
            model_endpoint=ModelEndpointRuntimeConfig(
                provider="local_gateway",
                model="timeline-static",
                reasoning_effort="high",
                base_url="http://127.0.0.1:9/v1",
                auth_mode="none",
                api_key_env="ASA_TEST_STATIC_TOKEN",
            ),
            agent_model_overrides=(),
        ),
        path,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
