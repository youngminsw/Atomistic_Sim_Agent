from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import parse_lammps_output_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = parse_lammps_output_run(
        run_dir=Path(args.fixture),
        material_id=args.material,
        descriptor_root=Path(args.descriptor_root),
        out_path=Path(args.out),
    )

    print(f"md_events_valid={str(report.ok).lower()}")
    print(f"event_count={report.event_count}")
    print(f"descriptors_present={str(report.descriptors_present).lower()}")
    print(f"layer_removed_count={report.layer_removed_count}")
    print(f"total_deposited_energy_eV={report.total_deposited_energy_eV:.1f}")
    if report.evidence:
        print(f"evidence={','.join(report.evidence)}")
    if report.errors:
        print(f"errors={','.join(report.errors)}")
    if report.output_path is not None:
        print(f"out={report.output_path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
