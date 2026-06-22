from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import KnowledgeRegistryError, ProvenanceRecord, SourceKind, seeded_provenance_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--title", default="Ad hoc provenance claim")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = ProvenanceRecord(
        record_id="dry-run-user-claim",
        source_url=args.source_url,
        title=args.title,
        claim=args.claim,
        tags=tuple(args.tag or ("user_import",)),
        confidence=0.5,
        extracted_on="2026-06-10",
        used_by=("knowledge",),
        source_kind=SourceKind.PAPER,
    )
    try:
        registry = seeded_provenance_registry().with_record(record)
    except KnowledgeRegistryError as exc:
        print(str(exc))
        return 1

    print("provenance_ready=true")
    print(f"dry_run={str(args.dry_run).lower()}")
    print("graphdb_write=false")
    print(f"source_count={len(registry.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
