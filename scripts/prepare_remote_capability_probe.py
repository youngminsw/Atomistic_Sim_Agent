from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import ComputePolicyError, prepare_remote_capability_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--remote-user")
    parser.add_argument("--ssh-target")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--requires-cuda", action="store_true")
    parser.add_argument("--requires-lammps", action="store_true")
    parser.add_argument("--required-lammps-packages", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        bundle = prepare_remote_capability_probe(
            source_root=SOURCE_ROOT,
            output_dir=Path(args.output_dir),
            host_alias=args.host,
            environment_name=args.environment_name,
            remote_user=args.remote_user,
            ssh_target=args.ssh_target,
            ssh_port=args.ssh_port,
            requires_cuda=args.requires_cuda,
            requires_lammps=args.requires_lammps,
            required_lammps_packages=_csv_tuple(args.required_lammps_packages),
        )
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("remote_capability_probe_ok=true")
    print(f"probe_script_path={bundle.script_path}")
    print(f"probe_manifest_path={bundle.manifest_path}")
    print(f"source_payload_path={bundle.source_payload_path}")
    print(f"run_command={bundle.manifest_payload['run_command']}")
    return 0


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
