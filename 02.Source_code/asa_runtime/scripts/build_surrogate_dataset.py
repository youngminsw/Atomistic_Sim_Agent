from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import verify_md_run
from sim_agent.ml_surrogate import KernelFeatureSpec, SurrogateDatasetError, build_training_dataset
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--required-ion")
    parser.add_argument("--required-material")
    args = parser.parse_args()

    try:
        report = verify_md_run(
            log_path=Path(args.log),
            events_path=Path(args.events),
            expected_events=args.expected_events,
            required_ion=args.required_ion,
            required_material=args.required_material,
        )
        spec = KernelFeatureSpec.from_mapping(
            as_mapping(json.loads(Path(args.kernel).read_text(encoding="utf-8")), "kernel")
        )
        dataset = build_training_dataset(report, spec)
    except (json.JSONDecodeError, OSError, SchemaValidationError, SurrogateDatasetError) as exc:
        print("surrogate_dataset_ok=false")
        print(str(exc))
        return 1

    print("surrogate_dataset_ok=true")
    print(f"kernel_id={dataset.kernel_id}")
    print(f"row_count={dataset.row_count}")
    print(f"feature_columns={','.join(dataset.feature_columns)}")
    print(f"total_removed_depth_nm={dataset.total_removed_depth_nm:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
