from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import (
    LAMMPSAssetStagingError,
    LAMMPSExecutionPlanError,
    LAMMPSInputDeckError,
    build_lammps_execution_plan,
    render_lammps_input_deck,
    stage_lammps_run_assets,
)
from sim_agent.md_campaign import MDCampaignStagingError, stage_md_campaign
from sim_agent.schemas._parse import JsonMap, as_mapping
from sim_agent.schemas.errors import SchemaValidationError


DEFAULT_MD_INCIDENT_COUNT = 500


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--contract-out", required=True)
    parser.add_argument("--incident-schedule-out", required=True)
    parser.add_argument("--surface-state-out", required=True)
    parser.add_argument("--lammps-input-out", required=True)
    parser.add_argument("--lammps-input-manifest-out", required=True)
    parser.add_argument("--lammps-assets-manifest-out", required=True)
    parser.add_argument("--lammps-execution-plan-out", required=True)
    parser.add_argument("--incident-count", type=int, default=DEFAULT_MD_INCIDENT_COUNT)
    parser.add_argument("--lammps-binary", default="lmp")
    args = parser.parse_args()

    try:
        campaign = _read_mapping(Path(args.plan), "md_campaign_plan")
        request = _read_mapping(Path(args.request), "validated_request")
        out_path = Path(args.out)
        contract_path = Path(args.contract_out)
        schedule_path = Path(args.incident_schedule_out)
        surface_state_path = Path(args.surface_state_out)
        input_path = Path(args.lammps_input_out)
        input_manifest_path = Path(args.lammps_input_manifest_out)
        assets_manifest_path = Path(args.lammps_assets_manifest_out)
        execution_plan_path = Path(args.lammps_execution_plan_out)
        staging = stage_md_campaign(
            campaign,
            request,
            Path(args.descriptor_root),
            incident_count=args.incident_count,
        )
        input_deck = render_lammps_input_deck(
            staging.lammps_contract_payload,
            staging.incident_schedule_payload,
            staging.surface_state_payload,
        )
        assets = stage_lammps_run_assets(
            staging.lammps_contract_payload,
            staging.surface_state_payload,
            input_path.parent,
            PROJECT_ROOT,
        )
        _write_json(out_path, staging.manifest_payload)
        _write_json(contract_path, staging.lammps_contract_payload)
        _write_json(schedule_path, staging.incident_schedule_payload)
        _write_json(surface_state_path, staging.surface_state_payload)
        _write_text(input_path, input_deck.input_script)
        _write_json(input_manifest_path, input_deck.manifest_payload)
        _write_json(assets_manifest_path, assets.manifest_payload)
        execution_plan = build_lammps_execution_plan(
            input_deck.manifest_payload,
            assets.manifest_payload,
            input_path.parent,
            lammps_binary=args.lammps_binary,
        )
        _write_json(execution_plan_path, execution_plan.manifest_payload)
    except (
        OSError,
        json.JSONDecodeError,
        SchemaValidationError,
        MDCampaignStagingError,
        LAMMPSAssetStagingError,
        LAMMPSExecutionPlanError,
        LAMMPSInputDeckError,
    ) as exc:
        print(str(exc))
        return 1

    print("md_campaign_job_ok=true")
    print(f"manifest_path={out_path}")
    print(f"contract_path={contract_path}")
    print(f"incident_schedule_path={schedule_path}")
    print(f"surface_state_path={surface_state_path}")
    print(f"lammps_input_path={input_path}")
    print(f"lammps_input_manifest_path={input_manifest_path}")
    print(f"lammps_assets_manifest_path={assets_manifest_path}")
    print(f"lammps_execution_plan_path={execution_plan_path}")
    return 0


def _read_mapping(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
