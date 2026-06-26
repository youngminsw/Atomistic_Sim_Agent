from __future__ import annotations

import argparse
from pathlib import Path
from typing import assert_never

from sim_agent.cli.auth import add_auth_parser, run_auth
from sim_agent.cli.adversarial_e2e_smoke import AdversarialE2ESmokeRequest, run_adversarial_e2e_smoke
from sim_agent.cli.compaction_smoke import CompactionSmokeRequest, run_compaction_smoke
from sim_agent.cli.e2e_runtime_smoke import E2ERuntimeSmokeError, E2ERuntimeSmokeRequest, run_e2e_runtime_smoke
from sim_agent.cli.orchestrator import add_chat_parser, run_chat
from sim_agent.cli.skill_workflow_smoke import SkillWorkflowSmokeRequest, run_skill_workflow_smoke
from sim_agent.cli.tui import run_tui
from sim_agent.cli.tui_control_room_smoke import TuiControlRoomSmokeRequest, run_tui_control_room_smoke
from sim_agent.cli.workflow_e2e_smoke import WorkflowE2ESmokeRequest, run_workflow_e2e_smoke
from sim_agent.cli.workflow_live_llm_e2e import WorkflowLiveLlmE2ERequest, run_workflow_live_llm_e2e
from sim_agent.ui import build_ui_api_status
from sim_agent.ui.server import build_ui_http_server


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.e2e_runtime_smoke:
        return _run_e2e_runtime_smoke(args)
    if args.skill_workflow_smoke:
        return _run_skill_workflow_smoke(args)
    if args.compaction_smoke:
        return _run_compaction_smoke(args)
    if args.tui_control_room_smoke:
        return _run_tui_control_room_smoke(args)
    if args.adversarial_e2e_smoke:
        return _run_adversarial_e2e_smoke(args)
    if args.workflow_e2e_smoke:
        return _run_workflow_e2e_smoke(args)
    if args.workflow_live_llm_e2e:
        return _run_workflow_live_llm_e2e(args)
    match args.command:
        case None:
            return run_tui(session_dir=Path(args.session_dir) if args.session_dir else None, resume=args.resume)
        case "chat":
            return run_chat(args)
        case "auth":
            return run_auth(args)
        case "ui":
            return _run_ui(args)
        case unreachable:
            assert_never(unreachable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asa",
        description="Atomistic Simulation Agent CLI",
        epilog="Run `asa` without a command to open the interactive agent shell.",
    )
    parser.add_argument("--session-dir", help="Open the interactive shell with a durable session directory.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume the latest interactive GlobalSession, or resume the supplied session id/path.",
    )
    parser.add_argument("--e2e-runtime-smoke", action="store_true", help="Run a narrow live ASA runtime smoke turn.")
    parser.add_argument("--skill-workflow-smoke", action="store_true", help="Run a markdown skill/workflow gate smoke.")
    parser.add_argument("--compaction-smoke", action="store_true", help="Run a semantic compaction/resume smoke.")
    parser.add_argument("--tui-control-room-smoke", action="store_true", help="Run a live PTY TUI control-room smoke.")
    parser.add_argument("--adversarial-e2e-smoke", action="store_true", help="Run adversarial runtime blocker smoke.")
    parser.add_argument("--workflow-e2e-smoke", action="store_true", help="Run canonical workflow command e2e smoke.")
    parser.add_argument(
        "--workflow-live-llm-e2e",
        action="store_true",
        help="Run live-LLM workflow e2e evidence, blocking when live providers are unavailable.",
    )
    parser.add_argument("--model-profile", default="codex-pro")
    parser.add_argument("--scenario", default="orchestrator_subagent_tool_loop")
    parser.add_argument("--allow-hardgate-bypass", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    subparsers = parser.add_subparsers(dest="command")
    add_chat_parser(subparsers)
    add_auth_parser(subparsers)
    ui = subparsers.add_parser("ui", help="Start the HTML controller.")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8779)
    ui.add_argument("--allow-non-loopback", action="store_true")
    ui.add_argument("--controller-token")
    ui.add_argument("--smoke", action="store_true")
    return parser


def _run_e2e_runtime_smoke(args: argparse.Namespace) -> int:
    if args.output_json is None:
        print("e2e_runtime_smoke_error=output_json_required")
        return 2
    try:
        evidence_path = run_e2e_runtime_smoke(
            E2ERuntimeSmokeRequest(
                model_profile=args.model_profile,
                scenario=args.scenario,
                allow_hardgate_bypass=args.allow_hardgate_bypass,
                output_json=args.output_json,
                session_dir=Path(args.session_dir) if args.session_dir else None,
            )
        )
    except E2ERuntimeSmokeError as exc:
        print(f"e2e_runtime_smoke_error={exc}")
        return 2
    print("e2e_runtime_smoke=true")
    print(f"e2e_runtime_smoke_json={evidence_path}")
    print("destructive_writes_ran=none")
    return 0


def _run_skill_workflow_smoke(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("skill_workflow_smoke_error=output_dir_required")
        return 2
    result = run_skill_workflow_smoke(SkillWorkflowSmokeRequest(output_dir=args.output_dir))
    print("skill_workflow_smoke=true")
    print(f"skill_workflow_smoke_status={result.status}")
    print(f"skill_parity_matrix={result.skill_matrix_path}")
    print(f"workflow_gate_parity_matrix={result.workflow_matrix_path}")
    print(f"skill_workflow_transcript={result.transcript_path}")
    for blocker in result.blockers:
        print(f"skill_workflow_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_compaction_smoke(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("compaction_smoke_error=output_dir_required")
        return 2
    result = run_compaction_smoke(CompactionSmokeRequest(output_dir=args.output_dir))
    print("compaction_smoke=true")
    print(f"compaction_smoke_status={result.status}")
    print(f"compaction_parity_matrix={result.matrix_path}")
    print(f"compaction_transcript={result.transcript_path}")
    print(f"compaction_e2e_surface={result.e2e_path}")
    for blocker in result.blockers:
        print(f"compaction_smoke_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_adversarial_e2e_smoke(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("adversarial_e2e_smoke_error=output_dir_required")
        return 2
    result = run_adversarial_e2e_smoke(AdversarialE2ESmokeRequest(output_dir=args.output_dir))
    print("adversarial_e2e_smoke=true")
    print(f"adversarial_e2e_smoke_status={result.status}")
    print(f"adversarial_e2e_smoke_json={result.output_json}")
    for blocker in result.blockers:
        print(f"adversarial_e2e_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_tui_control_room_smoke(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("tui_control_room_smoke_error=output_dir_required")
        return 2
    result = run_tui_control_room_smoke(TuiControlRoomSmokeRequest(output_dir=args.output_dir))
    print("tui_control_room_smoke=true")
    print(f"tui_control_room_smoke_status={result.status}")
    print(f"tui_command_parity_matrix={result.matrix_path}")
    print(f"tui_transcript={result.transcript_path}")
    print(f"tui_final_transcript={result.final_transcript_path}")
    for blocker in result.blockers:
        print(f"tui_control_room_smoke_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_workflow_e2e_smoke(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("workflow_e2e_smoke_error=output_dir_required")
        return 2
    result = run_workflow_e2e_smoke(
        WorkflowE2ESmokeRequest(output_dir=args.output_dir, scenario=args.scenario)
    )
    print("workflow_e2e_smoke=true")
    print(f"workflow_e2e_smoke_status={result.status}")
    print(f"workflow_e2e_smoke_json={result.output_json}")
    print(f"workflow_e2e_transcript={result.transcript_path}")
    for blocker in result.blockers:
        print(f"workflow_e2e_smoke_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_workflow_live_llm_e2e(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        print("workflow_live_llm_e2e_error=output_dir_required")
        return 2
    result = run_workflow_live_llm_e2e(
        WorkflowLiveLlmE2ERequest(output_dir=args.output_dir, scenario=args.scenario)
    )
    print("workflow_live_llm_e2e=true")
    print(f"workflow_live_llm_e2e_status={result.status}")
    print(f"workflow_live_llm_e2e_json={result.output_json}")
    print(f"workflow_live_llm_provider_events={result.provider_events_path}")
    for blocker in result.blockers:
        print(f"workflow_live_llm_e2e_blocker={blocker}")
    return 0 if result.status == "succeeded" else 1


def _run_ui(args: argparse.Namespace) -> int:
    status = build_ui_api_status()
    if args.smoke:
        _print_ui_smoke(status.static_root)
        return 0
    server = build_ui_http_server(
        args.host,
        args.port,
        status.static_root,
        allow_non_loopback=args.allow_non_loopback,
        csrf_token=args.controller_token,
    )
    with server:
        print(f"ui_url=http://{args.host}:{args.port}/run_bundle_viewer.html")
        server.serve_forever()
    return 0


def _print_ui_smoke(static_root: Path) -> None:
    status = build_ui_api_status()
    print("ui_smoke=true")
    print(f"static_root={static_root}")
    for route in status.routes:
        print(f"route={route.path}")


if __name__ == "__main__":
    raise SystemExit(main())
