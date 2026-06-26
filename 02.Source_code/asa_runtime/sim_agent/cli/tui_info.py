from __future__ import annotations

from typing import TextIO

from sim_agent.schemas._parse import as_str
from sim_agent.runtime_config import active_profile_status, load_runtime_config
from sim_agent.ui import build_ui_api_status

from .tui_paths import display_path
from .tui_render import write_command_palette, write_hud_panel, write_welcome
from .tui_state import TuiState, recent_events, resume_state
from .tui_timeline import write_timeline


def write_banner(state: TuiState, output_stream: TextIO) -> None:
    write_welcome(state, output_stream)


def write_help(output_stream: TextIO) -> None:
    write_command_palette("/", output_stream)
    output_stream.write("초보자 빠른 시작\n")
    output_stream.write("beginner_help=true\n")
    output_stream.write("beginner_first_step=/guide\n")
    output_stream.write("beginner_wizard=/wizard\n")
    output_stream.write("beginner_model_check=/model status\n")
    output_stream.write("beginner_runtime_test=/runtime tools\n")
    output_stream.write("beginner_goal_hint=그냥 목표를 한국어/영어 문장으로 입력해도 됩니다\n")
    output_stream.write("friendly_note=명령어를 외우지 않아도 됩니다. / 를 누르고 보이는 항목을 고르면 됩니다\n")
    output_stream.write("agent_mention_hint=@md_agent 처럼 입력하면 해당 AgentSession에 직접 전달합니다\n")
    output_stream.write("@md_agent MESSAGE, @qa_agent MESSAGE\n")
    output_stream.write("/chat [@agent] MESSAGE|clear  # transcript/control-room fallback\n")
    output_stream.write("/guide\n")
    output_stream.write("/start\n")
    output_stream.write("/wizard\n")
    output_stream.write("/model status|set|login\n")
    output_stream.write("/login [oauth|api-key] --provider <id>\n")
    output_stream.write("/hud\n")
    output_stream.write("/agents\n")
    output_stream.write("/compact [status|replay] [@agent|agent]\n")
    output_stream.write("/harness\n")
    output_stream.write("/workflow <name> [--gate-id ID] [--owner-agent AGENT] [--target-agent AGENT] [--output-dir PATH]\n")
    output_stream.write("/workflow-response <gate-id> <value> [--workflow-id NAME] [--responder-agent AGENT]\n")
    output_stream.write("/deep-interview | /ralplan | /ultrawork | /ultraqa | /ultragoal | /visual-qa | /ultraresearch\n")
    output_stream.write("/team [--output-dir PATH] [--simulate-agent-failure AGENT] [--slurm-job-script]\n")
    output_stream.write("/team contract\n")
    output_stream.write("/skills\n")
    output_stream.write("/tools\n")
    output_stream.write("/memory [live]\n")
    output_stream.write("/runtime tools\n")
    output_stream.write("/runtime [--output-dir PATH] [--smoke|--tool-gateway]\n")
    output_stream.write("/setup wizard|graphdb|endpoint|runtime\n")
    output_stream.write("/status\n")
    output_stream.write("/log [--limit N]\n")
    output_stream.write("/timeline [--limit N]\n")
    output_stream.write("/resume [latest|session_id|path]\n")
    output_stream.write("/run [--output-dir PATH] [--source-root PATH] <goal>\n")
    output_stream.write("/ui\n")
    output_stream.write("/exit\n")


def handle_status(state: TuiState, output_stream: TextIO) -> None:
    active = active_profile_status(load_runtime_config())
    profile_name = active.name or "none"
    output_stream.write("Session Status\n")
    output_stream.write("session_status=true\n")
    output_stream.write(f"session_id={state.session_id}\n")
    output_stream.write(f"session_dir={display_path(state.session_dir)}\n")
    output_stream.write(f"active_profile={profile_name}\n")
    output_stream.write(f"profile_customized={str(active.customized).lower()}\n")
    output_stream.write(f"model_profile={profile_name} customized={str(active.customized).lower()}\n")
    output_stream.write(f"model={state.model.provider}/{state.model.name}/{state.model.reasoning_effort}/{state.model.auth_mode}\n")
    output_stream.write(f"last_run_ledger={display_path(state.last_run_ledger)}\n")
    output_stream.write(f"team_ledger={display_path(state.team_ledger)}\n")
    output_stream.write(f"runtime_ledger={display_path(state.runtime_ledger)}\n")


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


def handle_timeline(limit: int, state: TuiState, output_stream: TextIO) -> None:
    write_timeline(state, output_stream, limit=limit)


def handle_resume(args: tuple[str, ...], state: TuiState, output_stream: TextIO) -> TuiState:
    target = args[0] if args else "latest"
    try:
        next_state = resume_state(state, target)
    except Exception as exc:  # noqa: BLE001 - command surface reports typed textual blocker.
        output_stream.write("resume=false\n")
        output_stream.write(f"resume_target={target}\n")
        output_stream.write(f"resume_error={exc}\n")
        return state
    output_stream.write("Session Resumed\n")
    output_stream.write("resume=true\n")
    output_stream.write(f"resume_target={target}\n")
    output_stream.write(f"session_id={next_state.session_id}\n")
    output_stream.write(f"session_dir={display_path(next_state.session_dir)}\n")
    return next_state


def handle_ui(output_stream: TextIO) -> None:
    status = build_ui_api_status()
    output_stream.write("HTML Controller\n")
    output_stream.write("ui_ready=true\n")
    output_stream.write("ui_command=asa ui --port 8779\n")
    output_stream.write(f"static_root={status.static_root}\n")
