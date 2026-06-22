from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.input_planner import plan_simulation_input
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()

    try:
        payload = as_mapping(json.loads(Path(args.fixture).read_text(encoding="utf-8")), "request")
    except (json.JSONDecodeError, OSError, SchemaValidationError) as exc:
        print("input_plan_ok=false")
        print(str(exc))
        return 1

    result = plan_simulation_input(payload)
    print("input_plan_ok=true")
    print(f"request_id={result.request_id}")
    print(f"mode={result.mode}")
    print(f"feature_type={result.feature_type}")
    print(f"geometry_kind={result.geometry_kind}")
    print(f"geometry_path={result.geometry_path}")
    print(f"geometry_units={result.geometry_units}")
    print(f"target_material={result.target_material_id}")
    print(f"mask_material={result.mask_material_id}")
    print(f"ion_species={result.ion_species}")
    print(f"clarification_required={str(result.clarification_required).lower()}")
    print(f"missing_fields={','.join(result.missing_fields)}")
    print(f"model_training_required={str(result.model_training_required).lower()}")
    print(f"training_reason={result.training_reason}")
    print(f"trained_kernel_id={result.trained_kernel_id}")
    for prompt in result.clarifications:
        print(f"clarification={prompt.field}:{prompt.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
