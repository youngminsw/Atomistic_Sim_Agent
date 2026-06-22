from __future__ import annotations

import argparse
from pathlib import Path

from sim_agent.cli.auth import add_auth_parser, run_auth
from sim_agent.cli.e2e_runtime_smoke import E2ERuntimeSmokeError, E2ERuntimeSmokeRequest, run_e2e_runtime_smoke
from sim_agent.cli.orchestrator import add_chat_parser, run_chat
from sim_agent.cli.tui import run_tui
from sim_agent.ui import build_ui_api_status
from sim_agent.ui.server import build_ui_http_server


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.e2e_runtime_smoke:
        return _run_e2e_runtime_smoke(args)
    match args.command:
        case None:
            return run_tui(session_dir=Path(args.session_dir) if args.session_dir else None, resume=args.resume)
        case "chat":
            return run_chat(args)
        case "auth":
            return run_auth(args)
        case "ui":
            return _run_ui(args)
        case _:
            parser.print_help()
            return 1


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
    parser.add_argument("--model-profile", default="codex-pro")
    parser.add_argument("--scenario", default="orchestrator_subagent_tool_loop")
    parser.add_argument("--allow-hardgate-bypass", action="store_true")
    parser.add_argument("--output-json", type=Path)
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
