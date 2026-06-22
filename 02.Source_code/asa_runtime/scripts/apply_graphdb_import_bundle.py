from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import (  # noqa: E402
    GraphDBAccessError,
    GraphDBWriteRequest,
    execute_graph_import_bundle,
    graphdb_write_report_payload,
)
from sim_agent.schemas._parse import JsonMap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--approve-write", action="store_true")
    parser.add_argument("--allow-non-empty", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    try:
        report = execute_graph_import_bundle(
            Path(args.bundle_dir),
            GraphDBWriteRequest(
                approve_write=args.approve_write,
                database_name=args.database_name,
                require_empty_database=not args.allow_non_empty,
            ),
        )
    except GraphDBAccessError as exc:
        print(f"graphdb_write_status=blocked")
        print(f"graphdb_write_blocker={exc}")
        if args.out:
            _write_json_report(
                Path(args.out),
                {
                    "applied": False,
                    "status": "blocked",
                    "blocker_reasons": [str(exc)],
                    "database_name": args.database_name,
                    "row_counts": {},
                    "executed_statement_kinds": [],
                },
            )
        return 2

    if args.out:
        _write_json_report(Path(args.out), graphdb_write_report_payload(report))
    for line in report.summary_lines():
        print(line)
    if report.applied:
        return 0
    return 2


def _write_json_report(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
