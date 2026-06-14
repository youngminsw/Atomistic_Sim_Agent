from __future__ import annotations

from typing import TextIO

from sim_agent.schemas._parse import as_str
from sim_agent.ui import build_ui_api_status

from .tui_render import write_command_palette, write_hud_panel, write_welcome
from .tui_state import TuiState, recent_events


def write_banner(state: TuiState, output_stream: TextIO) -> None:
    write_welcome(state, output_stream)


def write_help(output_stream: TextIO) -> None:
    write_command_palette("/", output_stream)
    output_stream.write("/model status|set|login\n")
    output_stream.write("/login [oauth|api-key] --provider <id>\n")
    output_stream.write("/hud\n")
    output_stream.write("/agents\n")
    output_stream.write("/harness\n")
    output_stream.write("/team [--output-dir PATH] [--simulate-agent-failure AGENT] [--slurm-job-script]\n")
    output_stream.write("/team contract\n")
    output_stream.write("/skills\n")
    output_stream.write("/runtime [--output-dir PATH] [--smoke]\n")
    output_stream.write("/status\n")
    output_stream.write("/log [--limit N]\n")
    output_stream.write("/run [--output-dir PATH] [--source-root PATH] <goal>\n")
    output_stream.write("/ui\n")
    output_stream.write("/exit\n")


def handle_status(state: TuiState, output_stream: TextIO) -> None:
    output_stream.write("Session Status\n")
    output_stream.write("session_status=true\n")
    output_stream.write(f"session_id={state.session_id}\n")
    output_stream.write(f"session_dir={state.session_dir}\n")
    output_stream.write(f"model={state.model.provider}/{state.model.name}/{state.model.auth_mode}\n")
    output_stream.write(f"last_run_ledger={state.last_run_ledger or ''}\n")
    output_stream.write(f"team_ledger={state.team_ledger or ''}\n")
    output_stream.write(f"runtime_ledger={state.runtime_ledger or ''}\n")


def handle_hud(state: TuiState, output_stream: TextIO) -> None:
    write_hud_panel(state, output_stream)


def handle_log(limit: int, state: TuiState, output_stream: TextIO) -> None:
    output_stream.write("Session Log\n")
    output_stream.write("session_log=true\n")
    for event in recent_events(state, limit):
        output_stream.write(
            f"event={as_str(event['event_type'], 'event_type')} "
            f"summary={as_str(event['summary'], 'summary')}\n"
        )


def handle_ui(output_stream: TextIO) -> None:
    status = build_ui_api_status()
    output_stream.write("HTML Controller\n")
    output_stream.write("ui_ready=true\n")
    output_stream.write("ui_command=asa ui --port 8779\n")
    output_stream.write(f"static_root={status.static_root}\n")
