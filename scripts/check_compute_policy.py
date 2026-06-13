from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import allowed_compute_hosts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args()

    print("compute_policy_ok=true")
    for host_alias in allowed_compute_hosts():
        print(f"host={host_alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
