from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import (
    ComputePolicyError,
    run_remote_capability_probe,
    write_remote_capability_probe_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    try:
        result = run_remote_capability_probe(
            manifest_path=Path(args.manifest),
            timeout_s=args.timeout_s,
        )
        write_remote_capability_probe_result(Path(args.out), result)
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("remote_capability_probe_runner_ok=true")
    print(f"probe_status={result.payload['probe_status']}")
    print(f"result_path={args.out}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
