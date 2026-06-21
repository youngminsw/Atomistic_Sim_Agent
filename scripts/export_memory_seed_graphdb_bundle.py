from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge.memory_seed import (  # noqa: E402
    DEFAULT_MEMORY_TERMS,
    MemorySeedError,
    build_memory_seed_bundle,
    read_memory_seed_sources_from_neo4j,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--database-name", default="atomistic_sim_agent_knowledge")
    parser.add_argument("--sync-run-id", default="personal-memory-seed")
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--limit-per-term", type=int, default=4)
    args = parser.parse_args()
    try:
        sources = read_memory_seed_sources_from_neo4j(
            terms=tuple(args.term or DEFAULT_MEMORY_TERMS),
            limit_per_term=max(1, args.limit_per_term),
        )
        bundle = build_memory_seed_bundle(
            Path(args.out),
            database_name=args.database_name,
            sync_run_id=args.sync_run_id,
            memory_sources=sources,
        )
    except MemorySeedError as exc:
        print(f"memory_seed_error={exc}")
        return 2
    print(f"memory_seed_source_count={len(sources)}")
    for line in bundle.summary_lines():
        print(line)
    return 0 if bundle.report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
