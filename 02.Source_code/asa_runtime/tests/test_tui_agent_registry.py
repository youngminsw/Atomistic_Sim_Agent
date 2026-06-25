from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.agent_runtime import load_agent_registry
from sim_agent.cli.tui_state import initial_state


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


EXPECTED_AGENT_IDS = (
    "orchestrator",
    "md_agent",
    "ml_agent",
    "feature_scale_agent",
    "research_agent",
    "qa_agent",
)


def test_initial_state_creates_persistent_agent_registry_and_handles(tmp_path: Path) -> None:
    state = initial_state(tmp_path)

    registry = load_agent_registry(state.session_dir)
    md_handle = registry.handles["md_agent"]

    assert tuple(registry.handles) == EXPECTED_AGENT_IDS
    assert registry.global_session_id == state.session_id
    assert md_handle.agent_id == "md_agent"
    assert md_handle.agent_session_id == f"{state.session_id}:md_agent"
    assert md_handle.session_dir == state.session_dir / "agent_sessions" / "md_agent"
    assert md_handle.messages_path == md_handle.session_dir / "messages.jsonl"
    assert md_handle.events_path == md_handle.session_dir / "events.jsonl"
    assert md_handle.model.name == state.model.name
    assert "physics" in md_handle.role_prompt.casefold()
    assert (state.session_dir / "agent_registry.json").is_file()
    assert (md_handle.session_dir / "session.json").is_file()


def test_agent_registry_resume_preserves_handle_and_appends_agent_event(tmp_path: Path) -> None:
    first = initial_state(tmp_path)
    first_registry = load_agent_registry(first.session_dir)
    first_md = first_registry.handles["md_agent"]

    resumed = initial_state(tmp_path, resume="latest")
    second_registry = load_agent_registry(resumed.session_dir)
    second_md = second_registry.handles["md_agent"]

    assert second_md.agent_session_id == first_md.agent_session_id
    assert second_md.created_at == first_md.created_at
    events = _jsonl(second_md.events_path)
    assert [event["event_type"] for event in events] == ["agent_session_registered", "agent_session_resumed"]
    assert [event["sequence"] for event in events] == [1, 2]


def test_agents_command_exposes_durable_agent_session_paths(tmp_path: Path) -> None:
    result = _run_module_interactive(["/agents", "/exit"], session_dir=tmp_path / "session")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_registry_path=" in result.stdout
    assert "agent_session=md_agent" in result.stdout
    assert "agent_session=qa_agent" in result.stdout
    assert "agent_session_mode=persistent" in result.stdout
    assert "auto_compaction_policy=llm_semantic_boundary_rewrite_active" in result.stdout
    assert "worker_adapter=disabled default_runtime_attached=false" in result.stdout


def _run_module_interactive(
    lines: list[str],
    *,
    session_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_STARTUP_WIZARD"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
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
