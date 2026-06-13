from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_cli_request import AgentCliRequestInput, build_agent_cli_request, parse_range
from sim_agent.agent_run_ledger import write_agent_run_ledger
from sim_agent.schemas._parse import as_str, require
from sim_agent.ui import agent_compute


SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class OrchestratorChatConfig:
    message: str
    output_dir: Path
    source_root: Path
    material: str
    phase: str
    ion: str
    feature_type: str
    mode: str
    energy_range_ev: str
    polar_range_deg: str
    azimuth_range_deg: str
    host: str
    environment_name: str
    model_provider: str
    model_name: str
    model_base_url: str
    model_auth_mode: str
    model_api_key_env: str


@dataclass(frozen=True, slots=True)
class OrchestratorChatReport:
    run_id: str
    artifact_dir: str
    ledger_path: Path


@dataclass(frozen=True, slots=True)
class OrchestratorChatError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def add_chat_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("chat", help="Talk to the main Orchestrator and prepare a run bundle.")
    parser.add_argument("--message", "-m", required=True)
    parser.add_argument("--output-dir", default=str(SOURCE_ROOT / "evidence" / "asa-chat"))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--material", default="Si")
    parser.add_argument("--phase", choices=("crystal", "amorphous"), default="amorphous")
    parser.add_argument("--ion", default="Ar")
    parser.add_argument("--feature-type", choices=("hole", "trench"), default="hole")
    parser.add_argument("--mode", choices=("2d", "3d"), default="3d")
    parser.add_argument("--energy-range-eV", default="30:150")
    parser.add_argument("--polar-range-deg", default="0:55")
    parser.add_argument("--azimuth-range-deg", default="0:360")
    parser.add_argument("--host", default="gpu-5090")
    parser.add_argument("--environment-name", default="atomistic-sim-gpu")
    parser.add_argument("--model-provider", default="openclaw")
    parser.add_argument("--model-name", default="gpt-5.5")
    parser.add_argument("--model-base-url", default="https://openclaw.local/v1")
    parser.add_argument("--model-auth-mode", choices=("api_key", "oauth", "gateway", "none"), default="oauth")
    parser.add_argument("--model-api-key-env", default="OPENCLAW_OAUTH_TOKEN")


def run_chat(args: argparse.Namespace) -> int:
    config = _config(args)
    try:
        report = prepare_orchestrator_chat(config)
    except OrchestratorChatError as exc:
        print(str(exc))
        return 1
    _print_report(report)
    return 0


def prepare_orchestrator_chat(config: OrchestratorChatConfig) -> OrchestratorChatReport:
    agent_compute.SOURCE_ROOT = config.source_root
    request = build_agent_cli_request(
        _request_input(config),
        parse_range(config.energy_range_ev, "energy_range_eV"),
        parse_range(config.polar_range_deg, "polar_range_deg"),
        parse_range(config.azimuth_range_deg, "azimuth_range_deg"),
    )
    response, status = agent_compute.build_agent_compute_bundle_http_response(
        {
            "request": request,
            "output_dir": str(config.output_dir),
            "host": config.host,
            "environment_name": config.environment_name,
            "md_incident_count": 500,
        }
    )
    if status != 200:
        raise OrchestratorChatError(as_str(require(response, "error"), "error"))
    ledger_path = write_agent_run_ledger(config.output_dir, request, response, None, None, None)
    return OrchestratorChatReport(
        run_id=as_str(require(response, "run_id"), "run_id"),
        artifact_dir=as_str(require(response, "artifact_dir"), "artifact_dir"),
        ledger_path=ledger_path,
    )


def _print_report(report: OrchestratorChatReport) -> None:
    print("asa_chat_ok=true")
    print("orchestrator=main")
    print(f"run_id={report.run_id}")
    print(f"artifact_dir={report.artifact_dir}")
    print(f"agent_run_ledger_path={report.ledger_path}")
    print("next=asa ui --port 8779")


def _config(args: argparse.Namespace) -> OrchestratorChatConfig:
    return OrchestratorChatConfig(
        message=args.message,
        output_dir=Path(args.output_dir),
        source_root=Path(args.source_root),
        material=args.material,
        phase=args.phase,
        ion=args.ion,
        feature_type=args.feature_type,
        mode=args.mode,
        energy_range_ev=args.energy_range_eV,
        polar_range_deg=args.polar_range_deg,
        azimuth_range_deg=args.azimuth_range_deg,
        host=args.host,
        environment_name=args.environment_name,
        model_provider=args.model_provider,
        model_name=args.model_name,
        model_base_url=args.model_base_url,
        model_auth_mode=args.model_auth_mode,
        model_api_key_env=args.model_api_key_env,
    )


def _request_input(config: OrchestratorChatConfig) -> AgentCliRequestInput:
    return AgentCliRequestInput(
        goal=config.message,
        material=config.material,
        phase=config.phase,
        ion=config.ion,
        feature_type=config.feature_type,
        mode=config.mode,
        flux_ions_cm2_s=1.0e15,
        active_layer_thickness_nm=1.0,
        pr_selectivity=20.0,
        md_box_x_nm=12.0,
        md_box_y_nm=12.0,
        md_box_mobile_depth_nm=9.0,
        md_box_fixed_depth_nm=1.0,
        md_box_thermostat_depth_nm=1.0,
        md_box_expected_cascade_depth_nm=6.0,
        md_box_atom_count=5000,
        md_timestep_fs=0.1,
        md_run_length_ps=2.0,
        lammps_structure_source=None,
        lammps_structure_preparation="user_supplied_relaxed_structure",
        model_provider=config.model_provider,
        model_name=config.model_name,
        model_base_url=config.model_base_url,
        model_auth_mode=config.model_auth_mode,
        model_api_key_env=config.model_api_key_env,
    )
