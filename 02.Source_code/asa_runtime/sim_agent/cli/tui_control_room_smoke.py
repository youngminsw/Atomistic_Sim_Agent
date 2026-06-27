from __future__ import annotations

import json
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Protocol

from sim_agent.runtime_config import (
    ActiveModelProfileRuntimeConfig,
    ModelEndpointRuntimeConfig,
    default_runtime_config,
    save_runtime_config,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class TuiControlRoomSmokeRequest:
    output_dir: Path


@dataclass(frozen=True, slots=True)
class TuiControlRoomSmokeResult:
    status: str
    matrix_path: Path
    transcript_path: Path
    final_transcript_path: Path
    blockers: tuple[str, ...]


def run_tui_control_room_smoke(request: TuiControlRoomSmokeRequest) -> TuiControlRoomSmokeResult:
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = output_dir / "task-11-tui-session"
    ctrl_c_session_dir = output_dir / "task-11-ctrlc-session"
    non_tty_session_dir = output_dir / "task-11-non-tty-session"
    for path in (session_dir, ctrl_c_session_dir, non_tty_session_dir):
        shutil.rmtree(path, ignore_errors=True)

    with _FakeProviderGateway() as gateway:
        config_path = _write_runtime_config(output_dir / "task-11-runtime-config.json", gateway.base_url)
        pty_transcript = _run_pty_control_room(session_dir, config_path)
        ctrl_c_transcript = _run_pty_ctrl_c(ctrl_c_session_dir, config_path)
        non_tty = _run_non_tty_control_room(non_tty_session_dir, config_path)

    combined = "\n".join(
        (
            "=== PTY CONTROL ROOM ===",
            pty_transcript.text,
            "=== PTY CTRL-C EXIT ===",
            ctrl_c_transcript.text,
            "=== NON-TTY FALLBACK ===",
            non_tty.stdout,
            "=== NON-TTY STDERR ===",
            non_tty.stderr,
        )
    )
    transcript_path = output_dir / "task-11-tui-transcript.txt"
    transcript_path.write_text(combined, encoding="utf-8")

    matrix = _matrix(
        pty_transcript=pty_transcript,
        ctrl_c_transcript=ctrl_c_transcript,
        non_tty=non_tty,
        combined=combined,
    )
    matrix_path = output_dir / "task-11-tui-command-parity-matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_transcript_path = output_dir / "final-f3-tui-transcript.txt"
    final_transcript_path.write_text(_final_transcript(combined, matrix), encoding="utf-8")
    blockers = tuple(str(item) for item in matrix["blockers"])
    return TuiControlRoomSmokeResult(
        status="succeeded" if not blockers else "blocked",
        matrix_path=matrix_path,
        transcript_path=transcript_path,
        final_transcript_path=final_transcript_path,
        blockers=blockers,
    )


@dataclass(frozen=True, slots=True)
class _PtyResult:
    text: str
    returncode: int | None
    cleanup: str


class _TerminableProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None:
        raise NotImplementedError

    def terminate(self) -> None:
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> int:
        raise NotImplementedError


def _run_pty_control_room(session_dir: Path, config_path: Path) -> _PtyResult:
    commands = (
        "/",
        "/model status",
        "/model thinking xhigh",
        "@md_agent 한글 MD control-room live check",
        "@qa_agent force artifact tool visual qa",
        "/hud",
        "/agents",
        "/tools",
        "/compact status",
        "/workflow catalog",
        "/exit",
    )
    return _run_pty(
        session_dir,
        config_path,
        commands=commands,
        startup_wizard=True,
        send_ctrl_c=False,
        columns=100,
        timeout_s=35,
    )


def _run_pty_ctrl_c(session_dir: Path, config_path: Path) -> _PtyResult:
    return _run_pty(
        session_dir,
        config_path,
        commands=(),
        startup_wizard=False,
        send_ctrl_c=True,
        columns=90,
        timeout_s=12,
    )


def _run_pty(
    session_dir: Path,
    config_path: Path,
    *,
    commands: tuple[str, ...],
    startup_wizard: bool,
    send_ctrl_c: bool,
    columns: int,
    timeout_s: float,
) -> _PtyResult:
    master_fd, slave_fd = pty.openpty()
    env = _env(session_dir, config_path, columns=columns)
    env["ASA_STARTUP_WIZARD"] = "1" if startup_wizard else "0"
    env["ASA_TUI_SEMANTIC"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "sim_agent"],
        cwd=SOURCE_ROOT,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    chunks: list[bytes] = []
    killed = False
    deadline = time.monotonic() + timeout_s
    try:
        if startup_wizard:
            _read_until(master_fd, chunks, ("ASA Wizard",), deadline)
            os.write(master_fd, b"\x1b")
        _read_until(master_fd, chunks, ("asa>",), deadline)
        if send_ctrl_c:
            os.write(master_fd, b"\x03")
            _read_until_exit(proc, master_fd, chunks, deadline)
        else:
            for command in commands:
                os.write(master_fd, command.encode("utf-8") + b"\r")
                if command == "/exit":
                    _read_until_exit(proc, master_fd, chunks, deadline)
                else:
                    _read_until(master_fd, chunks, ("asa>",), deadline)
        if proc.poll() is None:
            _read_until_exit(proc, master_fd, chunks, min(deadline, time.monotonic() + 1.5))
        if proc.poll() is None:
            killed = _stop_process(proc)
    finally:
        if proc.poll() is None:
            killed = _stop_process(proc) or killed
        _drain(master_fd, chunks, timeout_s=0.2)
        try:
            os.close(master_fd)
        except OSError:
            pass
    cleanup = f"pty_pid={proc.pid} exited={proc.poll() is not None} killed={str(killed).lower()}"
    return _PtyResult(_decode(chunks), proc.returncode, cleanup)


def _stop_process(proc: _TerminableProcess) -> bool:
    if proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=3)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        return True
    return True


def _read_until(master_fd: int, chunks: list[bytes], markers: tuple[str, ...], deadline: float) -> None:
    while time.monotonic() < deadline:
        text = _decode(chunks)
        if all(marker in text for marker in markers):
            return
        if not _read_once(master_fd, chunks, timeout_s=0.15):
            continue


def _read_until_exit(proc: subprocess.Popen[bytes], master_fd: int, chunks: list[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        _read_once(master_fd, chunks, timeout_s=0.1)
        if proc.poll() is not None:
            _drain(master_fd, chunks, timeout_s=0.2)
            return


def _drain(master_fd: int, chunks: list[bytes], *, timeout_s: float) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end and _read_once(master_fd, chunks, timeout_s=0.02):
        pass


def _read_once(master_fd: int, chunks: list[bytes], *, timeout_s: float) -> bool:
    try:
        readable, _writable, _error = select.select((master_fd,), (), (), timeout_s)
    except (OSError, ValueError):
        return False
    if not readable:
        return False
    try:
        data = os.read(master_fd, 4096)
    except OSError:
        return False
    if not data:
        return False
    chunks.append(data)
    return True


@dataclass(frozen=True, slots=True)
class _SubprocessResult:
    stdout: str
    stderr: str
    returncode: int
    max_box_width: int


def _run_non_tty_control_room(session_dir: Path, config_path: Path) -> _SubprocessResult:
    env = _env(session_dir, config_path, columns=80)
    env["ASA_STARTUP_WIZARD"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=SOURCE_ROOT,
        env=env,
        input="@qa_agent 한글 QA gate control-room fallback\n/chat\n/hud\n/exit\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=25,
    )
    return _SubprocessResult(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        max_box_width=_max_box_width(result.stdout),
    )


def _matrix(
    *,
    pty_transcript: _PtyResult,
    ctrl_c_transcript: _PtyResult,
    non_tty: _SubprocessResult,
    combined: str,
) -> dict[str, object]:
    checks = {
        "pty_returncode_zero": pty_transcript.returncode == 0,
        "startup_wizard_prompt": "ASA Wizard" in pty_transcript.text,
        "startup_wizard_esc_cancelled": "wizard_cancelled=true" in pty_transcript.text,
        "slash_palette": "Slash Command Palette" in pty_transcript.text and "/model" in pty_transcript.text,
        "direct_agent_activity_rail": "ASA Activity Rail" in pty_transcript.text and "agent_direct_route=md_agent" in pty_transcript.text,
        "tool_call_rows": "Tool Call" in pty_transcript.text and "tool_call=" in pty_transcript.text,
        "forced_tool_call_activity": (
            "agent_direct_route=qa_agent" in pty_transcript.text
            and "Tool Call" in pty_transcript.text
            and "artifact_write" in pty_transcript.text
            and "event=tool_start status=running detail=artifact_write" in pty_transcript.text
            and "event=tool_end status=ok detail=artifact_write" in pty_transcript.text
        ),
        "runtime_event_rows": "Runtime Events" in pty_transcript.text and "event=model_start" in pty_transcript.text,
        "chat_deck": "ASA Chat Deck" in pty_transcript.text,
        "hud": "ASA HUD" in pty_transcript.text and "control_room_hud=true" in pty_transcript.text,
        "agents_roster": "Persistent Domain Agents" in pty_transcript.text and "Global Bounded Subagent Presets" in pty_transcript.text,
        "model_state": "Model Status" in pty_transcript.text and "provider=openai-codex" in pty_transcript.text,
        "thinking_state": "Thinking level saved: xhigh" in pty_transcript.text and "thinking_level=xhigh" in pty_transcript.text,
        "login_state": "auth rail" in pty_transcript.text or "auth_mode=gateway" in pty_transcript.text,
        "tool_catalog": "Runtime Tool Catalog" in pty_transcript.text and "mcp_list_tools" in pty_transcript.text,
        "mcp_tool_row": "family=mcp" in pty_transcript.text and "mcp_call_tool" in pty_transcript.text,
        "compaction_row": "Compaction Status" in pty_transcript.text and "compact_agent=md_agent" in pty_transcript.text,
        "workflow_gate_row": "Workflow Catalog" in pty_transcript.text and "verification_gate=" in pty_transcript.text,
        "ctrl_c_exit": ctrl_c_transcript.returncode == 0 and "^C" in ctrl_c_transcript.text and "bye" in ctrl_c_transcript.text,
        "non_tty_returncode_zero": non_tty.returncode == 0,
        "non_tty_semantic_fallback": "agent_direct_route=qa_agent" in non_tty.stdout and "chat_window=true" in non_tty.stdout,
        "cjk_width_ok": 0 < non_tty.max_box_width <= 80,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "asa_tui_control_room_parity_v1",
        "status": "succeeded" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "pty": {
            "returncode": pty_transcript.returncode,
            "cleanup": pty_transcript.cleanup,
        },
        "ctrl_c": {
            "returncode": ctrl_c_transcript.returncode,
            "cleanup": ctrl_c_transcript.cleanup,
        },
        "non_tty": {
            "returncode": non_tty.returncode,
            "max_box_width": non_tty.max_box_width,
        },
        "transcript_chars": len(combined),
    }


def _final_transcript(combined: str, matrix: dict[str, object]) -> str:
    checks = matrix["checks"]
    assert isinstance(checks, dict)
    lines = [
        "ASA TUI Control Room Final Transcript",
        f"status={matrix['status']}",
        f"blockers={','.join(matrix['blockers']) if matrix['blockers'] else 'none'}",
    ]
    for key in sorted(checks):
        lines.append(f"check.{key}={checks[key]}")
    lines.append("")
    lines.append(combined)
    return "\n".join(lines)


def _env(session_dir: Path, config_path: Path, *, columns: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = str(columns)
    env["LINES"] = "30"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    env["ATOMISTIC_SIM_AGENT_PROVIDER_CREDENTIAL_STORE"] = str(config_path.parent / "task-11-credentials.json")
    env["ASA_TUI_SMOKE_TOKEN"] = "fake-tui-smoke-token"
    return env


def _write_runtime_config(path: Path, base_url: str) -> Path:
    default = default_runtime_config()
    config = replace(
        default,
        graphdb=default.graphdb,
        model_endpoint=ModelEndpointRuntimeConfig(
            provider="openai-codex",
            model="gpt-5.5",
            reasoning_effort="high",
            base_url=base_url,
            auth_mode="gateway",
            api_key_env="ASA_TUI_SMOKE_TOKEN",
        ),
        active_profile=ActiveModelProfileRuntimeConfig(name="codex-pro", customized=True),
    )
    return save_runtime_config(config, path)


class _FakeProviderGateway:
    def __init__(self) -> None:
        self._handler = type("_TuiFakeProviderGatewayHandler", (_FakeProviderGatewayHandler,), {"request_count": 0})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "_FakeProviderGateway":
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _FakeProviderGatewayHandler(BaseHTTPRequestHandler):
    request_count = 0

    def log_message(self, format: str, *args: str) -> None:
        return

    def do_POST(self) -> None:
        self.__class__.request_count += 1
        raw_body = self.rfile.read(int(self.headers.get("content-length", "0")))
        if self.path != "/v1/codex/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        payload = _json_payload(raw_body)
        if _is_forced_tool_request(payload) and not _has_tool_history(payload):
            self._write(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "artifact_write",
                            "arguments": json.dumps(
                                {
                                    "relative_path": "visual_qa/forced_tool_call.txt",
                                    "content": "forced artifact tool visual QA receipt",
                                }
                            ),
                        }
                    ]
                }
            )
            return
        self._write({"output_text": "ASA TUI live control room ok"})

    def _write(self, payload: dict[str, object], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _json_payload(raw_body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_forced_tool_request(payload: dict[str, object]) -> bool:
    return "force artifact tool visual qa" in json.dumps(payload, ensure_ascii=False)


def _has_tool_history(payload: dict[str, object]) -> bool:
    instructions = payload.get("instructions")
    return isinstance(instructions, str) and "[tool_history]" in instructions


def _decode(chunks: list[bytes]) -> str:
    return b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")


def _max_box_width(output: str) -> int:
    widths = [
        _cell_width(line)
        for line in output.splitlines()
        if line.startswith(("╭", "│", "├", "╰", "+", "|"))
    ]
    return max(widths, default=0)


def _cell_width(value: str) -> int:
    import unicodedata

    width = 0
    for char in value:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width
