from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sim_agent.runtime_config import default_runtime_config, save_runtime_config


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
ENTER = b"\n"
DOWN = b"\x1b[B"


@dataclass(frozen=True, slots=True)
class TuiProcess:
    process: subprocess.Popen[bytes]
    master_fd: int


def test_model_thinking_interactive_menu_accepts_arrow_selection(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/model thinking\n")
        output += _read_menu_until(session.master_fd, "fast lightweight subtasks")
        os.write(session.master_fd, DOWN + ENTER)
        output += _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/model status\n/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "Model Thinking Level" in output
    assert "model_thinking_level_saved=true" in output
    assert "thinking_level=xhigh reasoning_effort=xhigh" in output
    assert "provider=openai-codex model=gpt-5-codex reasoning_effort=xhigh" in output


def test_model_set_interactive_menu_asks_for_thinking_level(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/model set\n")
        output += _read_menu_until(session.master_fd, "local-open")
        os.write(session.master_fd, DOWN * 3 + ENTER)
        output += _read_menu_until(session.master_fd, "fast lightweight subtasks")
        os.write(session.master_fd, ENTER)
        output += _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "Model Selection" in output
    assert "Model Thinking Level" in output
    assert "model_saved=true" in output
    assert "provider=openai-codex model=gpt-5.5 reasoning_effort=xhigh" in output


def test_model_assign_interactive_menu_selects_agent_model_and_thinking(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/model assign\n")
        output += _read_menu_until(session.master_fd, "QA Agent")
        os.write(session.master_fd, ENTER)
        output += _read_menu_until(session.master_fd, "local-open")
        os.write(session.master_fd, ENTER)
        output += _read_menu_until(session.master_fd, "fast lightweight subtasks")
        os.write(session.master_fd, DOWN + DOWN + ENTER)
        output += _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/model agents\n/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "Agent Thinking Level" in output
    assert "agent_model_saved=md_agent" in output
    assert "agent=md_agent provider=openai-codex model=gpt-5.5 reasoning_effort=medium override=true" in output


def test_setup_endpoint_opens_interactive_endpoint_wizard(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/setup endpoint\n")
        output += _read_menu_until(session.master_fd, "Local gateway")
        os.write(session.master_fd, ENTER)
        output += _read_menu_until(session.master_fd, "fast lightweight subtasks")
        os.write(session.master_fd, ENTER)
        output += _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "Model Endpoint" in output
    assert "Model Thinking Level" in output
    assert "endpoint_config_saved=true" in output
    assert "provider=openai-codex model=gpt-5-codex reasoning_effort=high" in output


def test_setup_graphdb_opens_interactive_profile_wizard(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/setup graphdb\n")
        output += _read_menu_until(session.master_fd, "Local Neo4j")
        os.write(session.master_fd, ENTER)
        output += _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "GraphDB Profile" in output
    assert "graphdb_config_saved=true" in output
    assert "graphdb_uri=bolt://youngmin-lab:7687" in output
    assert "graphdb_database=atomistic_sim_agent_knowledge" in output


def test_run_without_args_opens_interview_run_wizard(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session_dir = tmp_path / "session"
    session = _spawn_tui(config_path, session_dir)

    try:
        output = _read_until(session.master_fd, "asa>")
        os.write(session.master_fd, b"/run\n")
        for marker in ("Simulation Goal", "Material", "Ion", "Phase", "Feature"):
            output += _read_menu_until(session.master_fd, marker)
            os.write(session.master_fd, ENTER)
        output += _read_until(session.master_fd, "asa>", timeout_s=30.0)
        os.write(session.master_fd, b"/exit\n")
        output += _read_until(session.master_fd, "bye")
    finally:
        _stop_tui(session)

    assert "deep_interview_prefill=true" in output
    assert "run_prepared=true" in output
    assert f"artifact_dir={session_dir}/wizard-run" in output


def _write_runtime_config(path: Path) -> Path:
    save_runtime_config(default_runtime_config(), path)
    return path


def _spawn_tui(config_path: Path, session_dir: Path) -> TuiProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    return TuiProcess(process=process, master_fd=master_fd)


def _read_until(fd: int, marker: str, timeout_s: float = 8.0) -> str:
    deadline = time.monotonic() + timeout_s
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        ready, _writable, _errors = select.select((fd,), (), (), 0.05)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        chunks.append(chunk)
        output = b"".join(chunks).decode("utf-8", errors="replace")
        if marker in output:
            return output
    output = b"".join(chunks).decode("utf-8", errors="replace")
    raise AssertionError(f"Timed out waiting for {marker!r}. Output:\n{output}")


def _read_menu_until(fd: int, marker: str) -> str:
    output = _read_until(fd, marker)
    return output + _read_until_idle(fd)


def _read_until_idle(fd: int, quiet_s: float = 0.12, timeout_s: float = 2.0) -> str:
    deadline = time.monotonic() + timeout_s
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        ready, _writable, _errors = select.select((fd,), (), (), quiet_s)
        if not ready:
            return b"".join(chunks).decode("utf-8", errors="replace")
        try:
            chunks.append(os.read(fd, 4096))
        except OSError:
            break
    return b"".join(chunks).decode("utf-8", errors="replace")


def _stop_tui(session: TuiProcess) -> None:
    if session.process.poll() is None:
        os.write(session.master_fd, b"/exit\n")
        session.process.terminate()
    session.process.wait(timeout=5)
    os.close(session.master_fd)
