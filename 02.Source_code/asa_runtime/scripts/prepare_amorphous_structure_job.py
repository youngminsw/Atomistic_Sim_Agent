from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import (
    AmorphousStructurePrepConfig,
    AmorphousStructurePrepError,
    stage_amorphous_structure_prep_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", required=True)
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lammps-binary", default="lmp")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    try:
        bundle = stage_amorphous_structure_prep_bundle(
            AmorphousStructurePrepConfig(
                material_id=args.material,
                atom_count=args.atom_count,
                lammps_binary=args.lammps_binary,
            ),
            Path(args.out_dir),
            PROJECT_ROOT,
        )
    except (AmorphousStructurePrepError, OSError) as exc:
        print(str(exc))
        return 1

    print("amorphous_structure_prep_ok=true")
    print(f"manifest_path={bundle.manifest_path}")
    print(f"structure_source_path={bundle.structure_source_path}")
    print(f"input_path={bundle.input_path}")
    print(f"potential_path={bundle.potential_path}")
    if args.execute:
        return _run_lammps(bundle.input_path.parent, args.lammps_binary)
    return 0


def _run_lammps(work_dir: Path, lammps_binary: str) -> int:
    completed = subprocess.run(
        [lammps_binary, "-in", "in.amorphous_prep"],
        cwd=work_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    output_structure = work_dir / "a_si_melt_quench_relaxed.data"
    payload = {
        "status": "passed" if completed.returncode == 0 and output_structure.exists() else "failed",
        "returncode": completed.returncode,
        "output_structure_path": str(output_structure),
        "output_structure_exists": output_structure.exists(),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    result_path = work_dir / "amorphous_structure_prep_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"execution_result_path={result_path}")
    if payload["status"] != "passed":
        return 1
    print("amorphous_structure_prep_execution_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
