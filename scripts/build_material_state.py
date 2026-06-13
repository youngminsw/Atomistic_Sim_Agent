from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.materials import MaterialBuilderError, build_material_state, material_report_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--phases")
    parser.add_argument("--method", default="fixture")
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--pr-selectivity", type=float, default=20.0)
    parser.add_argument("--out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        report = build_material_state(
            material_id=args.material,
            phases=_phases(args.phase, args.phases),
            descriptor_root=Path(args.descriptor_root),
            method=args.method,
            pr_selectivity=args.pr_selectivity,
        )
        descriptor_written = False
        if args.out:
            Path(args.out).write_text(json.dumps(material_report_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            descriptor_written = True
    except MaterialBuilderError as exc:
        print(str(exc))
        return 1

    print("material_state_ok=true")
    print(f"material={report.material_id}")
    print(f"dry_run={str(args.dry_run).lower()}")
    print(f"crystal_valid={str(report.crystal is not None).lower()}")
    print(f"amorphous_valid={str(report.amorphous is not None).lower()}")
    print(f"pr_selectivity={report.pr_material.selectivity}")
    print(f"force_field_protocol={report.force_field.protocol_id}")
    print(f"zbl_required={str(report.force_field.zbl_required).lower()}")
    print(f"descriptor_written={str(descriptor_written).lower()}")
    return 0


def _phases(phase: str | None, phases: str | None) -> tuple[str, ...]:
    value = phases or phase
    if value is None:
        return ("crystal",)
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
