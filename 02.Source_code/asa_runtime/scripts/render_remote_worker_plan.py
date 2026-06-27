from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import (
    ComputePolicyError,
    build_remote_execution_plan,
    load_worker_bundle,
    remote_execution_plan_manifest_payload,
)
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--out")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root")
    args = parser.parse_args()

    try:
        worker = load_worker_bundle(Path(args.worker))
        plan = build_remote_execution_plan(worker, ssh_target=args.ssh_target, ssh_port=args.ssh_port)
        output_root = _output_root(args.out, args.output_root)
        payload = remote_execution_plan_manifest_payload(
            plan,
            source_root=Path(args.source_root),
            output_root=output_root,
        )
        if args.out:
            _write_json(Path(args.out), payload)
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("remote_worker_plan_ok=true")
    print(f"ssh_target={plan.ssh_target}")
    print(f"ssh_port={plan.ssh_port}")
    for command in plan.all_commands:
        print(f"command={command}")
    return 0


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_root(out: str | None, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root)
    if out is None:
        return Path.cwd()
    out_path = Path(out)
    if out_path.parent.name == "remote":
        return out_path.parent.parent
    return out_path.parent


if __name__ == "__main__":
    raise SystemExit(main())
