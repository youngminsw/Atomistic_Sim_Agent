from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import MDVerificationError, verify_md_run
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--required-ion")
    parser.add_argument("--required-material")
    args = parser.parse_args()

    try:
        report = verify_md_run(
            log_path=Path(args.log),
            events_path=Path(args.events),
            expected_events=args.expected_events,
            required_ion=args.required_ion,
            required_material=args.required_material,
        )
    except (OSError, MDVerificationError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print(f"md_verified={str(report.ok).lower()}")
    print(f"status={report.status.value}")
    if report.dataset is not None:
        print(f"event_count={report.dataset.event_count}")
        print(f"total_deposited_energy_eV={report.dataset.total_deposited_energy_eV:.1f}")
        print(f"total_removed_depth_nm={report.dataset.total_removed_depth_nm:.3f}")
    if report.evidence:
        print(f"evidence={','.join(report.evidence)}")
    if report.errors:
        print(f"errors={','.join(report.errors)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
