from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import GraphDBGateError, GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--attempt-write", action="store_true")
    parser.add_argument("--user-db-approval", action="store_true")
    parser.add_argument("--existing-db", action="append", default=[])
    args = parser.parse_args()

    mode = GraphDBMode.ATTEMPT_WRITE if args.attempt_write else GraphDBMode.DRY_RUN
    request = GraphDBGateRequest(
        mode=mode,
        user_db_approval=args.user_db_approval,
        existing_database_names=tuple(args.existing_db),
    )
    try:
        plan = build_graphdb_gate_plan(request)
    except GraphDBGateError as exc:
        print(str(exc))
        return 1

    for line in plan.summary_lines():
        print(line)
    for item in plan.conflict_checks:
        print(f"conflict_check={item}")
    for item in plan.rollback_steps:
        print(f"rollback_step={item}")
    for item in plan.export_artifacts:
        print(f"export_artifact={item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
