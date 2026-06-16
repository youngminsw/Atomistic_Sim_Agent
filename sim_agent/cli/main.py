from __future__ import annotations

import argparse
from pathlib import Path

from sim_agent.cli.auth import add_auth_parser, run_auth
from sim_agent.cli.orchestrator import add_chat_parser, run_chat
from sim_agent.cli.tui import run_tui
from sim_agent.ui import build_ui_api_status
from sim_agent.ui.server import build_ui_http_server


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    match args.command:
        case None:
            return run_tui(session_dir=Path(args.session_dir) if args.session_dir else None)
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
