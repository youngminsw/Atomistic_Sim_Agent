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
    build_remote_execution_chain,
    load_worker_bundle,
    remote_execution_chain_payload,
)
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="append", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        workers = tuple(load_worker_bundle(Path(raw)) for raw in args.worker)
        chain = build_remote_execution_chain(
            workers,
            ssh_target=args.ssh_target,
            ssh_port=args.ssh_port,
        )
        payload = remote_execution_chain_payload(chain)
        _write_json(Path(args.out), payload)
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("remote_worker_chain_ok=true")
    print(f"ssh_target={chain.ssh_target}")
    print(f"ssh_port={chain.ssh_port}")
    print(f"stage_count={len(chain.stages)}")
    for stage in chain.stages:
        print(f"stage={stage.stage_id}")
    return 0


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
