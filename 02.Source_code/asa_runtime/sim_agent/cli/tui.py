from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import TextIO

from .tui_chat import handle_chat, handle_chat_message
from .tui_compaction import handle_compact
from .tui_info import handle_hud, handle_log, handle_resume, handle_status, handle_timeline, handle_ui, write_banner, write_help
from .tui_guide import handle_guide
from .tui_login import TerminalLoginSelector, handle_login
from .tui_memory import handle_memory
from .tui_model import handle_model
from .tui_parse import ParseError, parse_line, parse_options
from .tui_prompt import build_prompt_reader
from .tui_render import write_command_palette
from .tui_run import handle_run
from .tui_runtime import handle_runtime
from .tui_semantic import filter_semantic_tty_output
from .tui_setup import handle_setup
from .tui_state import TuiState, TuiStep, append_event, initial_state
from .tui_team import handle_agents, handle_contract, handle_harness, handle_skills, handle_team
from .tui_tools import handle_tools
from .tui_wizard import handle_wizard
from .tui_workflow import handle_workflow, write_workflow_catalog


def run_tui(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    *,
    session_dir: Path | None = None,
    resume: str | None = None,
) -> int:
    state = initial_state(session_dir, resume=resume)
    prompt_reader = build_prompt_reader(input_stream, output_stream)
    if prompt_reader.echoes_input:
        output_stream = filter_semantic_tty_output(output_stream)
        prompt_reader.output_stream = output_stream
    write_banner(state, output_stream)
    if prompt_reader.echoes_input and _startup_wizard_enabled():
        output_stream.write("startup_wizard=true\n")
        try:
            state = handle_wizard((), state, input_stream, output_stream, interactive=True)
        except KeyboardInterrupt:
            append_event(state, "session_exit", "User interrupted startup wizard")
            output_stream.write("^C\nbye\n")
            return 0
    while True:
        try:
            raw = prompt_reader.read_line()
        except KeyboardInterrupt:
            append_event(state, "session_exit", "User interrupted interactive shell")
            output_stream.write("^C\nbye\n")
            return 0
        if raw == "":
            output_stream.write("bye\n")
            return 0
        if not prompt_reader.echoes_input:
            output_stream.write("\n")
        try:
            step = _handle_line(raw.strip(), state, input_stream, output_stream, prompt_reader.echoes_input)
        except KeyboardInterrupt:
            append_event(state, "session_exit", "User interrupted interactive shell")
            output_stream.write("^C\nbye\n")
            return 0
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
    match parsed.command:  # noqa: MATCH_OK - user command strings are open-ended.
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
        case "/chat":
            return TuiStep(state=handle_chat(parsed.args, state, output_stream, handle_run), exit_requested=False)
        case "/guide" | "/start":
            handle_guide(state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/wizard":
            return TuiStep(
                state=handle_wizard(parsed.args, state, input_stream, output_stream, interactive=interactive),
                exit_requested=False,
            )
        case "/model":
            return TuiStep(state=handle_model(parsed.args, state, output_stream, input_stream), exit_requested=False)
        case "/login":
            selector = TerminalLoginSelector(input_stream, output_stream) if interactive else None
            return TuiStep(state=handle_login(parsed.args, state, output_stream, selector), exit_requested=False)
        case "/hud":
            handle_hud(state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/run":
            if interactive and not parsed.args:
                return TuiStep(
                    state=handle_wizard(("interview_run",), state, input_stream, output_stream, interactive=True),
                    exit_requested=False,
                )
            return TuiStep(state=handle_run(parsed.args, state, output_stream), exit_requested=False)
        case "/agents":
            handle_agents(state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/compact":
            return TuiStep(state=handle_compact(parsed.args, state, output_stream), exit_requested=False)
        case "/harness":
            handle_harness(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/workflow":
            if parsed.args and parsed.args[0] == "catalog":
                write_workflow_catalog(output_stream)
                return TuiStep(state=state, exit_requested=False)
            return TuiStep(state=handle_workflow(parsed.args, state, output_stream), exit_requested=False)
        case "/deep-interview" | "/ralplan" | "/ultrawork" | "/ultraqa" | "/ultragoal":
            workflow_id = parsed.command.removeprefix("/")
            return TuiStep(state=handle_workflow((workflow_id, *parsed.args), state, output_stream), exit_requested=False)
        case "/skills":
            handle_skills(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/tools":
            handle_tools(output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/team":
            return TuiStep(state=_handle_team(parsed.args, state, output_stream), exit_requested=False)
        case "/runtime":
            return TuiStep(state=handle_runtime(parsed.args, state, output_stream), exit_requested=False)
        case "/memory" | "/graphdb":
            handle_memory(parsed.args, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/setup":
            if parsed.args and parsed.args[0] == "wizard":
                return TuiStep(
                    state=handle_wizard(parsed.args[1:], state, input_stream, output_stream, interactive=interactive),
                    exit_requested=False,
                )
            if interactive and parsed.args in {(), ("endpoint",), ("graphdb",)}:
                return TuiStep(
                    state=handle_wizard(parsed.args, state, input_stream, output_stream, interactive=True),
                    exit_requested=False,
                )
            return TuiStep(state=handle_setup(parsed.args, state, output_stream), exit_requested=False)
        case "/status":
            handle_status(state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/log":
            handle_log(_log_limit(parsed.args), state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/timeline":
            handle_timeline(_log_limit(parsed.args), state, output_stream)
            return TuiStep(state=state, exit_requested=False)
        case "/resume":
            return TuiStep(state=handle_resume(parsed.args, state, output_stream), exit_requested=False)
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
    if not args and command.casefold() in {"cancel", "stop", "abort"}:
        append_event(state, "input_cancelled", command)
        output_stream.write("cancelled=true\n")
        return TuiStep(state=state, exit_requested=False)
    return TuiStep(state=handle_chat_message((command, *args), state, output_stream, handle_run), exit_requested=False)


def _log_limit(args: tuple[str, ...]) -> int:
    parsed = parse_options(args)
    value = parsed.options.get("limit", "20")
    if value.isdecimal():
        return int(value)
    return 20


def _startup_wizard_enabled() -> bool:
    value = os.environ.get("ASA_STARTUP_WIZARD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}
