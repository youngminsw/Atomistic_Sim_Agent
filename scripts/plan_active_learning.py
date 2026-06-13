from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md_campaign import ActiveLearningPlanError, plan_active_learning_run
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        plan = plan_active_learning_run(Path(args.run))
    except (ActiveLearningPlanError, SchemaValidationError) as exc:
        print("active_learning_plan_ok=false")
        print(str(exc))
        return 1

    print("active_learning_plan_ok=true")
    print(f"run_id={plan.run_id}")
    print(f"same_expert={str(plan.same_expert).lower()}")
    print(f"batch_size={plan.batch_size}")
    print(f"controlled_event_probe_allowed={str(plan.controlled_event_probe_allowed).lower()}")
    for request in plan.requests:
        print(f"request_id={request.request_id}")
        print(f"protocol={request.protocol}")
        print(f"handoff_target={request.handoff_target}")
        print(f"force_field_protocol_id={request.force_field_protocol_id}")
        print(f"physics_scope={request.physics_scope}")
        print(f"sample_count={request.sample_count}")
        print(f"sample_ids={','.join(request.sample_ids)}")
        print(f"uncertainty_threshold={request.uncertainty_threshold}")
        print(f"same_material_only={str(request.same_material_only).lower()}")
        print(f"energy_range_eV={request.energy_range_eV[0]}:{request.energy_range_eV[1]}")
        print(f"polar_range_deg={request.polar_range_deg[0]}:{request.polar_range_deg[1]}")
        print(f"snapshot_reset_required={str(request.snapshot_reset_required).lower()}")
        print(f"source_snapshot_ids={','.join(request.source_snapshot_ids)}")
    print(f"dry_run={str(args.dry_run).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
