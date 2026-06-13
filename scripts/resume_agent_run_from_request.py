from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agent_run_ledger import write_agent_run_ledger
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.ui.agent_compute import build_agent_compute_bundle_http_response


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume a planned agent run after a relaxed structure source exists."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--lammps-structure-source", required=True)
    parser.add_argument(
        "--lammps-structure-preparation",
        default="melt_quench_relaxed_structure",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--remote-user", default="swym")
    parser.add_argument("--ssh-target")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--md-incident-count", type=int, default=500)
    args = parser.parse_args()

    try:
        request = _request_with_structure_source(
            Path(args.request),
            args.lammps_structure_source,
            args.lammps_structure_preparation,
        )
        output_dir = Path(args.output_dir)
        response, status = build_agent_compute_bundle_http_response(
            {
                "request": request,
                "output_dir": str(output_dir),
                "host": args.host,
                "environment_name": args.environment_name,
                "remote_user": args.remote_user,
                "ssh_target": args.ssh_target,
                "ssh_port": args.ssh_port,
                "md_incident_count": args.md_incident_count,
            }
        )
        if status != 200:
            print(as_str(require(response, "error"), "error"))
            return 1
        ledger_path = write_agent_run_ledger(
            output_dir,
            request,
            response,
            amorphous_prep_result_path=None,
            capability_result_path=None,
            chain_result_path=None,
            surrogate_gate_result_path=None,
        )
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc))
        return 1

    print("resume_agent_run_ok=true")
    print(f"validated_request_path={response['validated_request_path']}")
    print(f"agent_run_ledger_path={ledger_path}")
    print(f"remote_execution_manifest_path={response.get('remote_execution_manifest_path', '')}")
    return 0


def _request_with_structure_source(
    request_path: Path,
    structure_source: str,
    preparation: str,
) -> JsonMap:
    request = dict(
        as_mapping(
            json.loads(request_path.read_text(encoding="utf-8")),
            "simulation_request",
        )
    )
    scene = dict(as_mapping(require(request, "scene"), "scene"))
    surface = dict(as_mapping(require(scene, "surface_state"), "surface_state"))
    phase = as_str(require(surface, "phase"), "surface_state.phase")
    surface["lammps_structure_source"] = {
        "kind": "user_supplied",
        "path": structure_source,
        "phase": phase,
        "preparation": preparation,
    }
    scene["surface_state"] = surface
    request["scene"] = scene
    return request


if __name__ == "__main__":
    raise SystemExit(main())
