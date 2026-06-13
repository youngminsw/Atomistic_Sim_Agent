from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.materials import MaterialBuilderError, build_material_state
from sim_agent.md import LAMMPSContractError, build_lammps_output_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--run-id", default="validated-md-run")
    args = parser.parse_args()

    try:
        material = build_material_state(
            material_id=args.material,
            phases=("crystal",),
            descriptor_root=Path(args.descriptor_root),
            method="fixture",
            pr_selectivity=20.0,
        )
        contract = build_lammps_output_contract(args.run_id, material.force_field)
    except (MaterialBuilderError, LAMMPSContractError) as exc:
        print("contract_valid=false")
        print(str(exc))
        return 1

    report = contract.validate_output_dir(Path(args.run_dir))
    print(f"contract_valid={str(report.ok).lower()}")
    print(f"run_id={contract.run_id}")
    print(f"units={contract.unit_system.unit_style}")
    print(f"energy_unit={contract.unit_system.energy_unit}")
    print(f"zbl_required={str(contract.collision_treatment.zbl_required).lower()}")
    for line in report.error_lines:
        print(line)
    print(f"missing_files={','.join(report.missing_filenames)}")
    print(f"found_files={','.join(report.found_filenames)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
