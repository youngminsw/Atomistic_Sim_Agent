from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import (  # noqa: E402
    GraphDBConnectionConfig,
    GraphDBGateError,
    GraphDBGateRequest,
    GraphDBMode,
    ResearchQuestion,
    build_graphdb_gate_plan,
    build_research_graphdb_agent_artifacts,
    seeded_provenance_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--existing-db", action="append", default=[])
    parser.add_argument("--database-name", default="atomistic_sim_agent_knowledge")
    parser.add_argument("--sync-run-id", default="research-graphdb-agent-sync")
    parser.add_argument("--query", default="What source-backed knowledge should agents read before simulation?")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--neo4j-uri", default="bolt://youngmin-lab:7687")
    parser.add_argument("--neo4j-user-env", default="NEO4J_USER")
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    args = parser.parse_args()

    try:
        gate_plan = build_graphdb_gate_plan(
            GraphDBGateRequest(
                mode=GraphDBMode.DRY_RUN,
                user_db_approval=False,
                existing_database_names=tuple(args.existing_db),
                database_name=args.database_name,
                requires_empty_database=True,
            )
        )
    except GraphDBGateError as exc:
        print(str(exc))
        return 1

    result = build_research_graphdb_agent_artifacts(
        seeded_provenance_registry(),
        gate_plan,
        Path(args.out),
        sync_run_id=args.sync_run_id,
        question=ResearchQuestion(query=args.query, tags=tuple(args.tag or ())),
        connection=GraphDBConnectionConfig(
            uri=args.neo4j_uri,
            database_name=args.database_name,
            username_env=args.neo4j_user_env,
            password_env=args.neo4j_password_env,
        ),
    )
    for line in result.summary_lines():
        print(line)
    for blocker in result.bundle.report.blocker_reasons:
        print(f"graphdb_ingest_blocker={blocker}")
    return 0 if result.status == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
