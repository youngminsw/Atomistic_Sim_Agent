from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import seeded_provenance_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()

    registry = seeded_provenance_registry()
    records = registry.list_by_tag(args.tag) if args.tag else registry.records
    missing_url = any(not record.source_url for record in records)
    print(f"source_count={len(records)}")
    print(f"missing_url={str(missing_url).lower()}")
    for record in records:
        print(f"{record.record_id}\t{','.join(record.tags)}\t{record.source_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
