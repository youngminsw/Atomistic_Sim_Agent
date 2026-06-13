from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.knowledge import ResearchImportRequest, ResearchToolError, import_research_source, seeded_provenance_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="")
    parser.add_argument("--claim", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    request = ResearchImportRequest(
        source_url=args.url,
        title=args.title or _default_title(args.url),
        claim=args.claim or _default_claim(args.tag),
        tags=tuple(args.tag or ("research_import",)),
        used_by=("knowledge",),
        record_id=_default_record_id(args.url),
    )
    try:
        result = import_research_source(seeded_provenance_registry(), request)
    except ResearchToolError as exc:
        print(str(exc))
        return 1

    print(f"provenance_ready={str(result.provenance_ready).lower()}")
    print(f"dry_run={str(args.dry_run).lower()}")
    print(f"graphdb_write={str(result.graphdb_write).lower()}")
    print(f"record_id={result.imported_record.record_id}")
    print(f"record_count={len(result.registry.records)}")
    return 0


def _default_title(source_url: str) -> str:
    if "sethian.etch3.pdf" in source_url:
        return "Level set methods for etching, deposition, and lithography development"
    return "Imported research source"


def _default_claim(tags: list[str]) -> str:
    if "level_set" in tags:
        return "Feature profile evolution can be represented by interface motion driven by local velocity fields."
    return "Imported source-backed claim requires review before physics acceptance."


def _default_record_id(source_url: str) -> str:
    if "sethian.etch3.pdf" in source_url:
        return "sethian-etch3"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
