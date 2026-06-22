from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.ui import build_ui_api_status
from sim_agent.ui.server import build_ui_http_server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-fixtures", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    status = build_ui_api_status()
    if args.smoke:
        _print_smoke(status.static_root)
        return 0
    _serve_static_ui(args.host, args.port, status.static_root)
    return 0


def _print_smoke(static_root: Path) -> None:
    status = build_ui_api_status()
    print("ui_smoke=true")
    print(f"static_root={static_root.relative_to(PROJECT_ROOT).as_posix()}")
    for route in status.routes:
        print(f"route={route.path}")
    for fixture in status.offline_fixtures:
        print(f"fixture={fixture}")
    print(f"auth_modes={','.join(status.auth_modes)}")
    print(f"model_providers={','.join(status.model_providers)}")


def _serve_static_ui(host: str, port: int, static_root: Path) -> None:
    server = build_ui_http_server(host, port, static_root)
    with server:
        print(f"ui_url=http://{host}:{port}/run_bundle_viewer.html")
        server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
