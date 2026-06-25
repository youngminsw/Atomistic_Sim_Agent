from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agent_cli_request import (
    AgentCliRequestInput,
    build_agent_cli_request,
    parse_range,
)
from sim_agent.agent_run_ledger import write_agent_run_ledger
from sim_agent.agent_cli_remote_actions import (
    AgentCliRemoteActionConfig,
    run_requested_remote_actions,
)
from sim_agent.compute import ComputePolicyError
from sim_agent.provider_registry import OPENAI_CODEX_BASE_URL, OPENAI_CODEX_TOKEN_ENV
from sim_agent.schemas._parse import JsonMap, as_str, require
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.ui.agent_compute import (
    build_agent_compute_bundle_http_response,
)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    try:
        if not args.offline:
            raise SchemaValidationError(
                "only --offline agent planning is wired in this build slice"
            )
        energy_range = parse_range(args.energy_range_eV, "energy_range_eV")
        polar_range = parse_range(args.polar_range_deg, "polar_range_deg")
        azimuth_range = parse_range(args.azimuth_range_deg, "azimuth_range_deg")
        request = build_agent_cli_request(
            _request_input(args),
            energy_range,
            polar_range,
            azimuth_range,
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
    except SchemaValidationError as exc:
        print(str(exc))
        return 1

    if status != 200:
        print(as_str(require(response, "error"), "error"))
        return 1

    _print_success(response, request)
    try:
        remote_result = run_requested_remote_actions(
            _remote_action_config(args),
            response,
            output_dir,
        )
        ledger_path = write_agent_run_ledger(
            output_dir,
            request,
            response,
            remote_result.amorphous_prep_result_path,
            remote_result.capability_result_path,
            remote_result.chain_result_path,
            remote_result.surrogate_gate_path,
        )
        print(f"agent_run_ledger_path={ledger_path}")
        return remote_result.exit_code
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the Simulation Agent to prepare a production pipeline run bundle."
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--phase", choices=("crystal", "amorphous"), required=True)
    parser.add_argument("--ion", required=True)
    parser.add_argument("--feature-type", choices=("hole", "trench"), required=True)
    parser.add_argument("--mode", choices=("2d", "3d"), default="3d")
    parser.add_argument("--energy-range-eV", required=True)
    parser.add_argument("--polar-range-deg", required=True)
    parser.add_argument("--azimuth-range-deg", required=True)
    parser.add_argument("--flux-ions-cm2-s", type=float, default=1.0e15)
    parser.add_argument("--active-layer-thickness-nm", type=float, default=1.0)
    parser.add_argument("--pr-selectivity", type=float, default=20.0)
    parser.add_argument("--md-incident-count", type=int, default=500)
    parser.add_argument("--md-box-x-nm", type=float, default=12.0)
    parser.add_argument("--md-box-y-nm", type=float, default=12.0)
    parser.add_argument("--md-box-mobile-depth-nm", type=float, default=9.0)
    parser.add_argument("--md-box-fixed-depth-nm", type=float, default=1.0)
    parser.add_argument("--md-box-thermostat-depth-nm", type=float, default=1.0)
    parser.add_argument("--md-box-expected-cascade-depth-nm", type=float, default=6.0)
    parser.add_argument("--md-box-atom-count", type=int, default=5000)
    parser.add_argument("--md-timestep-fs", type=float, default=0.1)
    parser.add_argument("--md-run-length-ps", type=float, default=2.0)
    parser.add_argument("--lammps-structure-source")
    parser.add_argument(
        "--lammps-structure-preparation",
        default="user_supplied_relaxed_structure",
    )
    parser.add_argument("--model-provider", default="openai-codex")
    parser.add_argument("--model-name", default="gpt-5-codex")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--model-base-url", default=OPENAI_CODEX_BASE_URL)
    parser.add_argument("--model-auth-mode", choices=("api_key", "oauth", "gateway", "none"), default="oauth")
    parser.add_argument("--model-api-key-env", default=OPENAI_CODEX_TOKEN_ENV)
    parser.add_argument("--host", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--remote-user", default="swym")
    parser.add_argument("--ssh-target")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--run-amorphous-structure-prep", action="store_true")
    parser.add_argument("--run-remote-capability-probe", action="store_true")
    parser.add_argument("--run-remote-chain", action="store_true")
    parser.add_argument("--remote-run-timeout-s", type=float)
    parser.add_argument("--surrogate-training-gate-report")
    parser.add_argument("--output-dir", required=True)
    return parser


def _request_input(args: argparse.Namespace) -> AgentCliRequestInput:
    return AgentCliRequestInput(
        goal=args.goal,
        material=args.material,
        phase=args.phase,
        ion=args.ion,
        feature_type=args.feature_type,
        mode=args.mode,
        flux_ions_cm2_s=args.flux_ions_cm2_s,
        active_layer_thickness_nm=args.active_layer_thickness_nm,
        pr_selectivity=args.pr_selectivity,
        md_box_x_nm=args.md_box_x_nm,
        md_box_y_nm=args.md_box_y_nm,
        md_box_mobile_depth_nm=args.md_box_mobile_depth_nm,
        md_box_fixed_depth_nm=args.md_box_fixed_depth_nm,
        md_box_thermostat_depth_nm=args.md_box_thermostat_depth_nm,
        md_box_expected_cascade_depth_nm=args.md_box_expected_cascade_depth_nm,
        md_box_atom_count=args.md_box_atom_count,
        md_timestep_fs=args.md_timestep_fs,
        md_run_length_ps=args.md_run_length_ps,
        lammps_structure_source=args.lammps_structure_source,
        lammps_structure_preparation=args.lammps_structure_preparation,
        model_provider=args.model_provider,
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
        model_base_url=args.model_base_url,
        model_auth_mode=args.model_auth_mode,
        model_api_key_env=args.model_api_key_env,
    )


def _print_success(response: JsonMap, request: JsonMap) -> None:
    print("agent_cli_ok=true")
    print(f"request_id={as_str(require(request, 'request_id'), 'request_id')}")
    print(f"run_id={as_str(require(response, 'run_id'), 'run_id')}")
    print("pipeline_stage=agent_plan")
    print("pipeline_stage=md_campaign_worker_bundle")
    print("pipeline_stage=lammps_execution_worker_bundle")
    print("pipeline_stage=md_postprocess_worker_bundle")
    print(f"artifact_dir={as_str(require(response, 'artifact_dir'), 'artifact_dir')}")
    plan_path = as_str(require(response, "md_campaign_plan_path"), "md_campaign_plan_path")
    lammps_path = as_str(
        require(response, "lammps_execution_worker_path"),
        "lammps_execution_worker_path",
    )
    print(f"md_campaign_plan_path={plan_path}")
    print(f"lammps_execution_worker_path={lammps_path}")
    _print_optional_path(response, "amorphous_structure_prep_manifest_path")
    _print_optional_path(response, "amorphous_structure_source_path")
    _print_optional_path(response, "amorphous_structure_prep_job_path")
    _print_optional_path(response, "amorphous_structure_prep_worker_path")
    _print_optional_path(response, "amorphous_structure_prep_remote_plan_path")
    _print_optional_path(response, "remote_execution_chain_path")
    _print_optional_path(response, "remote_execution_script_path")
    _print_optional_path(response, "remote_execution_manifest_path")
    _print_optional_path(response, "graphdb_agent_report_path")
    _print_optional_path(response, "graphdb_import_bundle_dir")
    _print_optional_path(response, "graphdb_ingest_report_path")


def _print_optional_path(response: JsonMap, field: str) -> None:
    value = response.get(field)
    if isinstance(value, str) and value:
        print(f"{field}={value}")


def _remote_action_config(args: argparse.Namespace) -> AgentCliRemoteActionConfig:
    return AgentCliRemoteActionConfig(
        source_root=SOURCE_ROOT,
        host=args.host,
        environment_name=args.environment_name,
        remote_user=args.remote_user,
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        remote_run_timeout_s=args.remote_run_timeout_s,
        run_amorphous_structure_prep=args.run_amorphous_structure_prep,
        run_remote_capability_probe=args.run_remote_capability_probe,
        run_remote_chain=args.run_remote_chain,
        surrogate_training_gate_report=args.surrogate_training_gate_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
