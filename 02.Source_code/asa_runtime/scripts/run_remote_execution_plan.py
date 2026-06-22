from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import (
    ComputePolicyError,
    run_remote_execution_plan,
    write_remote_execution_plan_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-s", type=float)
    args = parser.parse_args()

    try:
        result = run_remote_execution_plan(Path(args.plan), args.timeout_s)
        write_remote_execution_plan_result(Path(args.out), result)
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("remote_execution_plan_runner_ok=true")
    print(f"plan_status={result.payload['plan_status']}")
    print(f"result_path={args.out}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
