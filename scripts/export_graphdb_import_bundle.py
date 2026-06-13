from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import (  # noqa: E402
    GraphDBGateError,
    GraphDBGateRequest,
    GraphDBMode,
    build_graphdb_gate_plan,
    build_source_graph_import_bundle,
    seeded_provenance_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--attempt-write", action="store_true")
    parser.add_argument("--user-db-approval", action="store_true")
    parser.add_argument("--existing-db", action="append", default=[])
    parser.add_argument("--database-name", default="atomistic_sim_agent_knowledge")
    parser.add_argument("--sync-run-id", default="demo-empty-graphdb-sync")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mode = GraphDBMode.ATTEMPT_WRITE if args.attempt_write else GraphDBMode.DRY_RUN
    request = GraphDBGateRequest(
        mode=mode,
        user_db_approval=args.user_db_approval,
        existing_database_names=tuple(args.existing_db),
        database_name=args.database_name,
        requires_empty_database=True,
    )
    try:
        plan = build_graphdb_gate_plan(request)
    except GraphDBGateError as exc:
        print(str(exc))
        return 1

    bundle = build_source_graph_import_bundle(
        seeded_provenance_registry(),
        plan,
        Path(args.out),
        sync_run_id=args.sync_run_id,
    )
    for line in bundle.summary_lines():
        print(line)
    for blocker in bundle.report.blocker_reasons:
        print(f"graphdb_ingest_blocker={blocker}")
    return 0 if bundle.report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
