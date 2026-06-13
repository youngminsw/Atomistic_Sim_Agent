from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md_campaign import (
    DEFAULT_FORCE_FIELD_PROTOCOL_ID,
    DEFAULT_PHYSICS_SCOPE,
    CampaignPlanError,
    ExpertReference,
    MDCampaignPlan,
    plan_md_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", required=True)
    parser.add_argument("--ion", required=True)
    parser.add_argument("--phases", default="crystal")
    parser.add_argument("--iedf", default="20:200")
    parser.add_argument("--iadf", default="0:60")
    parser.add_argument("--azimuth", default="0:360")
    parser.add_argument("--extend-expert")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        plan = plan_md_campaign(
            material_id=args.material,
            ion_species=args.ion,
            phases=_phases(args.phases),
            energy_range_eV=_range(args.iedf),
            polar_range_deg=_range(args.iadf),
            azimuth_range_deg=_range(args.azimuth),
            extend_expert=_expert_reference(args.extend_expert),
        )
    except CampaignPlanError as exc:
        print("md_campaign_plan_ok=false")
        print(str(exc))
        return 1

    _print_plan(plan, args.dry_run)
    return 0


def _print_plan(plan: MDCampaignPlan, dry_run: bool) -> None:
    print("md_campaign_plan_ok=true")
    print(f"{plan.protocol_id}=true")
    print(f"event_probe_default={str(plan.event_probe_default).lower()}")
    print(f"material={plan.material_id}")
    print(f"ion={plan.ion_species}")
    print(f"phases={','.join(plan.phases)}")
    print(f"force_field_protocol={plan.force_field_protocol_id}")
    print(f"physics_scope={plan.physics_scope}")
    print(_strata_line("energy_strata_eV", plan.energy_strata.minimum, plan.energy_strata.maximum))
    print(_strata_line("polar_strata_deg", plan.polar_strata.minimum, plan.polar_strata.maximum))
    print(_strata_line("azimuth_strata_deg", plan.azimuth_strata.minimum, plan.azimuth_strata.maximum))
    print(f"pre_state_descriptors={','.join(plan.pre_state_descriptors)}")
    print(
        "layer_renewal="
        f"{plan.layer_renewal.renewal_action}@{plan.layer_renewal.removed_depth_threshold_nm}nm"
    )
    print(f"dry_run={str(dry_run).lower()}")


def _range(raw: str) -> tuple[float, float]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise CampaignPlanError("invalid_range")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise CampaignPlanError("invalid_range") from exc


def _phases(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _expert_reference(raw: str | None) -> ExpertReference | None:
    if raw is None:
        return None
    parts = raw.split("_on_")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise CampaignPlanError("invalid_expert_id")
    return ExpertReference(
        expert_id=raw,
        ion_species=parts[0],
        material_id=parts[1],
        force_field_protocol_id=DEFAULT_FORCE_FIELD_PROTOCOL_ID,
        physics_scope=DEFAULT_PHYSICS_SCOPE,
    )


def _strata_line(label: str, minimum: float, maximum: float) -> str:
    return f"{label}={minimum}:{maximum}"


if __name__ == "__main__":
    raise SystemExit(main())
