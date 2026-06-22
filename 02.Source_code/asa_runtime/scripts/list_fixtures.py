from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.testing.fixture_inventory import REQUIRED_FIXTURES, load_fixture, resolve_fixture, validate_required_fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-required", action="store_true")
    parser.add_argument("--fixture")
    args = parser.parse_args()

    if args.fixture:
        path = resolve_fixture(SOURCE_ROOT, args.fixture)
        if not path.exists():
            print(f"fixture_not_found={args.fixture}")
            return 1
        print(f"fixture_found={path}")

    report = validate_required_fixtures(SOURCE_ROOT)
    print(f"fixtures_ok={str(report.ok).lower()}")
    print(f"fixture_count={len(REQUIRED_FIXTURES)}")
    print(f"missing_count={len(report.missing)}")
    print(f"invalid_count={len(report.invalid)}")
    print(f"loaded_kinds={','.join(sorted(report.loaded_kinds))}")

    for missing in report.missing:
        print(f"missing={missing}")
    for invalid in report.invalid:
        print(f"invalid={invalid}")

    if args.assert_required and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
