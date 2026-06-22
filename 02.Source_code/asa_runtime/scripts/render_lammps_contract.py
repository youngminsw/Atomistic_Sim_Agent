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
    parser.add_argument("--material", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
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
        print(str(exc))
        return 1

    print("lammps_contract_ok=true")
    print("required_outputs_ok=true")
    print(f"run_id={contract.run_id}")
    print(f"units={contract.unit_system.unit_style}")
    print(f"energy_unit={contract.unit_system.energy_unit}")
    print(f"distance_unit={contract.unit_system.distance_unit}")
    print(f"zbl_required={str(contract.collision_treatment.zbl_required).lower()}")
    print(f"force_field_protocol={contract.collision_treatment.force_field_protocol_id}")
    print(f"required_outputs={','.join(contract.required_filenames)}")
    print(f"dry_run={str(args.dry_run).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
