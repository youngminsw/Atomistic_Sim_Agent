from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from .tui_info import handle_log, handle_status, handle_ui, write_banner, write_help
from .tui_login import TerminalLoginSelector, handle_login
from .tui_model import handle_model
from .tui_parse import ParseError, parse_line, parse_options
from .tui_prompt import build_prompt_reader
from .tui_render import write_command_palette
from .tui_run import handle_run
from .tui_runtime import handle_runtime
from .tui_state import TuiState, TuiStep, append_event, initial_state
from .tui_team import handle_agents, handle_contract, handle_harness, handle_skills, handle_team


def run_tui(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    *,
    session_dir: Path | None = None,
) -> int:
    state = initial_state(session_dir)
    prompt_reader = build_prompt_reader(input_stream, output_stream)
    write_banner(state, output_stream)
    while True:
        raw = prompt_reader.read_line()
        if raw == "":
            output_stream.write("bye\n")
            return 0
        if not prompt_reader.echoes_input:
            output_stream.write("\n")
        step = _handle_line(raw.strip(), state, input_stream, output_stream, prompt_reader.echoes_input)
        state = step.state
        if step.exit_requested:
            return 0


def _handle_line(
    line: str,
    state: TuiState,
    input_stream: TextIO,
    output_stream: TextIO,
    interactive: bool,
) -> TuiStep:
    try:
        parsed = parse_line(line)
    except ParseError as exc:
        output_stream.write(f"command_error={exc}\n")
        return TuiStep(state=state, exit_requested=False)
    if parsed is None:
        return TuiStep(state=state, exit_requested=False)
    match parsed.command:
        case "/":
            write_command_palette("/", output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/exit" | "/quit":
            append_event(state, "session_exit", "User closed interactive shell")
            output_stream.write("bye\n")
            return TuiStep(state=state, exit_requested=True)
        case "/help":
            write_help(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/model":
            return TuiStep(state=handle_model(parsed.args, state, output_stream), exit_requested=False)
        case "/login":
            selector = TerminalLoginSelector(input_stream, output_stream) if interactive else None
            return TuiStep(state=handle_login(parsed.args, state, output_stream, selector), exit_requested=False)
        case "/run":
            return TuiStep(state=handle_run(parsed.args, state, output_stream), exit_requested=False)
        case "/agents":
            handle_agents(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/harness":
            handle_harness(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/skills":
            handle_skills(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/team":
            return TuiStep(state=_handle_team(parsed.args, state, output_stream), exit_requested=False)
        case "/runtime":
            return TuiStep(state=handle_runtime(parsed.args, state, output_stream), exit_requested=False)
        case "/status":
            handle_status(state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/log":
            handle_log(_log_limit(parsed.args), state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/ui":
            handle_ui(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case _:
            return _handle_default(parsed.command, parsed.args, state, output_stream)


def _handle_team(args: tuple[str, ...], state: TuiState, output_stream: TextIO) -> TuiState:
    if args and args[0] == "contract":
        handle_contract(output_stream)
        return state
    return handle_team(args, state, output_stream)


def _handle_default(
    command: str,
    args: tuple[str, ...],
    state: TuiState,
    output_stream: TextIO,
) -> TuiStep:
    if command.startswith("/"):
        output_stream.write(f"unknown_command={command}\n")
        write_command_palette(command, output_stream)
        return TuiStep(state=state, exit_requested=False)
    return TuiStep(state=handle_run((command, *args), state, output_stream), exit_requested=False)


def _log_limit(args: tuple[str, ...]) -> int:
    parsed = parse_options(args)
    value = parsed.options.get("limit", "20")
    if value.isdecimal():
        return int(value)
    return 20
