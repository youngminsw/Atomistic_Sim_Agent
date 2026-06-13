from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import ComputePolicyError, build_worker_bundle, load_job_bundle, worker_bundle_payload
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--remote-user", default="swym")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    try:
        job = load_job_bundle(Path(args.job))
        bundle = build_worker_bundle(args.host, job, remote_user=args.remote_user)
    except ComputePolicyError as exc:
        print(str(exc))
        return 1

    if not args.dry_run:
        print("remote_execution_not_implemented")
        return 1

    if args.out:
        _write_json(Path(args.out), worker_bundle_payload(bundle))

    print("worker_bundle_ok=true")
    print(f"host={bundle.host_alias}")
    print(f"environment_name={bundle.environment_name}")
    print(f"remote_run_dir={bundle.remote_run_dir}")
    print(f"command_line={bundle.command_line}")
    for item in bundle.preflight_commands:
        print(f"preflight={item}")
    for item in bundle.input_paths:
        print(f"input={item}")
    for item in bundle.output_paths:
        print(f"output={item}")
    for item in bundle.transfer_plan:
        print(f"transfer={item}")
    return 0


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
