from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.cli.tui_direct_agent import DirectAgentChatRequest, run_direct_agent_chat
from sim_agent.cli.tui_state import initial_state


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_canonical_domain_agent_mentions_route_to_persistent_sessions(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@ml_agent audit the surrogate gate\n@research_agent check sources\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_direct_route=ml_agent" in result.stdout
    assert "agent_direct_route=research_agent" in result.stdout
    assert (session_dir / "agent_sessions" / "ml_agent" / "messages.jsonl").is_file()
    assert (session_dir / "agent_sessions" / "research_agent" / "messages.jsonl").is_file()


def test_unknown_domain_agent_mentions_are_not_direct_agent_targets(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@surrogate_worker audit the gate\n@graph_worker check sources\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_direct_route=surrogate_worker" not in result.stdout
    assert "agent_direct_route=graph_worker" not in result.stdout
    assert not (session_dir / "agent_sessions" / "surrogate_worker" / "messages.jsonl").exists()
    assert not (session_dir / "agent_sessions" / "graph_worker" / "messages.jsonl").exists()


def test_deprecated_domain_agent_mentions_do_not_create_sessions(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@deprecated_ml_worker audit the surrogate gate\n@deprecated_research_worker check sources\n/agents\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_direct_route=deprecated_ml_worker" not in result.stdout
    assert "agent_direct_route=deprecated_research_worker" not in result.stdout
    assert "agent_session=deprecated_ml_worker" not in result.stdout
    assert "agent_session=deprecated_research_worker" not in result.stdout
    assert not (session_dir / "agent_sessions" / "deprecated_ml_worker").exists()
    assert not (session_dir / "agent_sessions" / "deprecated_research_worker").exists()


def test_lower_level_direct_agent_dispatch_blocks_deprecated_target(tmp_path: Path) -> None:
    state = initial_state(tmp_path)

    result = run_direct_agent_chat(
        DirectAgentChatRequest(
            target="deprecated_ml_worker",
            message="do not create a deprecated session",
            session_id=state.session_id,
            session_dir=state.session_dir,
        )
    )

    assert result.turn_status == "blocked"
    assert result.agent_session_id == ""
    assert result.selected_tools == ()
    assert not (state.session_dir / "agent_sessions" / "deprecated_ml_worker").exists()


def test_resume_preserves_direct_domain_agent_session_transcript(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_dir = tmp_path / "session"
    first = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@md_agent first persisted turn\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )
    second = subprocess.run(
        [sys.executable, "-m", "sim_agent", "--resume"],
        cwd=PROJECT_ROOT,
        env=env,
        input="@md_agent second persisted turn\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    messages = _jsonl(session_dir / "agent_sessions" / "md_agent" / "messages.jsonl")
    events = _jsonl(session_dir / "agent_sessions" / "md_agent" / "events.jsonl")
    user_messages = [message for message in messages if message["role"] == "user"]
    assert [message["content"] for message in user_messages] == [
        "first persisted turn",
        "second persisted turn",
    ]
    assert len({message["agent_session_id"] for message in messages}) == 1
    assert sum(1 for event in events if event["event_type"] == "agent_session_resumed") >= 1


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    env["ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE"] = str(tmp_path / "credentials.json")
    env.pop("MODEL_GATEWAY_TOKEN", None)
    return env


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
