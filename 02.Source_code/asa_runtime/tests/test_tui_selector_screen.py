from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from sim_agent.runtime_config import default_runtime_config, save_runtime_config

pytest.importorskip("termios")
import pty


SOURCE_ROOT: Final = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Final = SOURCE_ROOT
ENTER: Final = b"\r"
DOWN: Final = b"\x1b[B"
CSI_PATTERN: Final = re.compile(r"\x1b\[([?0-9;]*)([A-Za-z])")


@dataclass(frozen=True, slots=True)
class TuiProcess:
    process: subprocess.Popen[bytes]
    master_fd: int
    argv: tuple[str, ...]


class TuiReadTimeout(AssertionError):
    def __init__(self, marker: str, timeout_s: float, output: str) -> None:
        super().__init__(f"Timed out waiting for {marker!r}. Output:\n{output}")
        self.marker = marker
        self.timeout_s = timeout_s
        self.output = output


@dataclass(frozen=True, slots=True)
class PtyDiagnostic:
    argv: tuple[str, ...]
    key_sequence: tuple[str, ...]
    timeout_s: float
    marker: str
    transcript: str
    process_status: str
    cleanup_status: str
    expected_failure_signature: str


def test_wizard_nested_selectors_replace_previous_menu(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/wizard"))
        output += _read_menu_until(session.master_fd, "return to shell")
        os.write(session.master_fd, DOWN + ENTER)
        output += _read_menu_until(session.master_fd, "Login Company")
        company_screen = _rendered_screen(output)
        os.write(session.master_fd, ENTER)
        output += _read_menu_until(session.master_fd, "Login Provider")
        provider_screen = _rendered_screen(output)
        os.write(session.master_fd, b"\x1b")
        output += _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/exit"))
        _read_until(session.master_fd, "bye")
        _wait_for_exit(session)
    finally:
        _stop_tui(session)

    assert "Login Company" in company_screen
    assert "ASA Wizard" not in company_screen
    assert "Model endpoint" not in company_screen
    assert "Login Provider" in provider_screen
    assert "Login Company" not in provider_screen
    assert "ASA Wizard" not in provider_screen


def test_model_thinking_menu_can_select_xhigh(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")

    try:
        output = _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/model thinking"))
        output += _read_menu_until(session.master_fd, "xhigh")
        os.write(session.master_fd, DOWN + ENTER)
        output += _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/model status") + _command("/exit"))
        output += _read_until(session.master_fd, "bye")
        _wait_for_exit(session)
    finally:
        _stop_tui(session)

    assert "Model Thinking Level" in output
    assert "model_thinking_level_saved=true" in output
    assert "thinking_level=xhigh reasoning_effort=xhigh" in output
    assert "provider=openai-codex model=gpt-5-codex reasoning_effort=xhigh" in output


def test_tui_selector_timeout_repro_has_diagnostic(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    session = _spawn_tui(config_path, tmp_path / "session")
    key_sequence = ("/wizard", "ENTER", "ESC", "/exit", "ENTER")
    expected_signature = "wizard selector should render after one terminal Enter"
    output = ""
    failure: TuiReadTimeout | None = None
    process_status = "not_observed"

    try:
        output = _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/wizard"))
        output += _read_menu_until(session.master_fd, "return to shell", timeout_s=4.0)
        os.write(session.master_fd, b"\x1b")
        output += _read_prompt(session.master_fd)
        os.write(session.master_fd, _command("/exit"))
        output += _read_until(session.master_fd, "bye")
        _wait_for_exit(session)
        process_status = _process_status(session)
    except TuiReadTimeout as exc:
        failure = exc
        process_status = _process_status(session)
    finally:
        cleanup_status = _stop_tui(session)

    if failure is not None:
        raise AssertionError(
            _format_pty_diagnostic(
                PtyDiagnostic(
                    argv=session.argv,
                    key_sequence=key_sequence,
                    timeout_s=failure.timeout_s,
                    marker=failure.marker,
                    transcript=failure.output,
                    process_status=process_status,
                    cleanup_status=cleanup_status,
                    expected_failure_signature=expected_signature,
                )
            )
        ) from failure

    _write_optional_diagnostic(
        PtyDiagnostic(
            argv=session.argv,
            key_sequence=key_sequence,
            timeout_s=4.0,
            marker="bye",
            transcript=output,
            process_status=process_status,
            cleanup_status=cleanup_status,
            expected_failure_signature="selector accepted intended key sequence and exited cleanly",
        )
    )
    assert "ASA Wizard" in output
    assert "return to shell" in output
    assert "bye" in output
    assert cleanup_status == "already_exited:returncode=0"


def _rendered_screen(output: str) -> str:
    rows = [""]
    row = 0
    column = 0
    index = 0
    while index < len(output):
        char = output[index]
        if char == "\x1b":
            match = CSI_PATTERN.match(output, index)
            if match is None:
                index += 1
                continue
            params, command = match.groups()
            if command == "A":
                row = max(0, row - _first_csi_int(params, 1))
            elif command == "C":
                column += _first_csi_int(params, 1)
            elif command == "D":
                column = max(0, column - _first_csi_int(params, 1))
            elif command == "K" and params == "2":
                rows[row] = ""
            elif command == "J":
                rows[row] = rows[row][:column]
                del rows[row + 1 :]
            index = match.end()
        elif char == "\r":
            column = 0
            index += 1
        elif char == "\n":
            row += 1
            column = 0
            _ensure_row(rows, row)
            index += 1
        else:
            rows[row] = _write_char(rows[row], column, char)
            column += 1
            index += 1
    return "\n".join(line.rstrip() for line in rows)


def _write_char(line: str, column: int, char: str) -> str:
    if column > len(line):
        line = f"{line}{' ' * (column - len(line))}"
    if column == len(line):
        return f"{line}{char}"
    return f"{line[:column]}{char}{line[column + 1:]}"


def _ensure_row(rows: list[str], row: int) -> None:
    while len(rows) <= row:
        rows.append("")


def _first_csi_int(params: str, default: int) -> int:
    first = params.split(";", 1)[0].lstrip("?")
    if not first.isdecimal():
        return default
    return int(first)


def _write_runtime_config(path: Path) -> Path:
    save_runtime_config(default_runtime_config(), path)
    return path


def _command(value: str) -> bytes:
    return value.encode("utf-8") + ENTER


def _spawn_tui(config_path: Path, session_dir: Path) -> TuiProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ASA_TUI_SEMANTIC"] = "1"
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    master_fd, slave_fd = pty.openpty()
    argv = (sys.executable, "-m", "sim_agent")
    process = subprocess.Popen(
        argv,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    return TuiProcess(process=process, master_fd=master_fd, argv=argv)


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
    raise TuiReadTimeout(marker, timeout_s, output)


def _read_prompt(fd: int) -> str:
    return _read_until(fd, "asa>") + _read_until_idle(fd)


def _read_menu_until(fd: int, marker: str, timeout_s: float = 8.0) -> str:
    return _read_until(fd, marker, timeout_s=timeout_s) + _read_until_idle(fd)


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


def _stop_tui(session: TuiProcess) -> str:
    cleanup_status = "already_exited"
    if session.process.poll() is None:
        cleanup_status = "terminated_after_exit_request"
        try:
            os.write(session.master_fd, _command("/exit"))
        except OSError:
            cleanup_status = "terminated_after_closed_pty"
        session.process.terminate()
    try:
        session.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        cleanup_status = f"{cleanup_status}:killed_after_wait_timeout"
        session.process.kill()
        session.process.wait(timeout=5)
    os.close(session.master_fd)
    return f"{cleanup_status}:returncode={session.process.returncode}"


def _wait_for_exit(session: TuiProcess, timeout_s: float = 2.0) -> None:
    try:
        session.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"TUI process did not exit after bye; status={_process_status(session)}") from exc


def _process_status(session: TuiProcess) -> str:
    return "running" if session.process.poll() is None else f"exited:returncode={session.process.returncode}"


def _format_pty_diagnostic(diagnostic: PtyDiagnostic) -> str:
    transcript = diagnostic.transcript or "<empty>"
    screen = _rendered_screen(diagnostic.transcript).strip() or "<empty>"
    return "\n".join(
        (
            "PTY selector diagnostic",
            f"pty_argv={list(diagnostic.argv)!r}",
            f"key_sequence={list(diagnostic.key_sequence)!r}",
            f"timeout_budget_s={diagnostic.timeout_s}",
            f"expected_marker={diagnostic.marker!r}",
            f"expected_failure_signature={diagnostic.expected_failure_signature}",
            f"process_status={diagnostic.process_status}",
            f"cleanup_status={diagnostic.cleanup_status}",
            "last_screen_snapshot:",
            screen,
            "raw_transcript:",
            transcript,
        )
    )


def _write_optional_diagnostic(diagnostic: PtyDiagnostic) -> None:
    path = os.environ.get("ASA_TUI_SELECTOR_DIAGNOSTIC_PATH")
    if path is None:
        return
    Path(path).write_text(f"{_format_pty_diagnostic(diagnostic)}\n", encoding="utf-8")
