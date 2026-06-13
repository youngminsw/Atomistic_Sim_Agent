from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.runner import OfflineRunRequest, run_offline_simulation


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene")
    source.add_argument("--image")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--ions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    mode = "3d" if args.scene is not None else "2d"
    result = run_offline_simulation(
        OfflineRunRequest(
            run_id=args.run_id,
            mode=mode,
            source_root=SOURCE_ROOT,
            output_dir=Path(args.out),
            scene_path=Path(args.scene) if args.scene is not None else None,
            image_path=Path(args.image) if args.image is not None else None,
            kernel_path=Path(args.kernel),
            events_path=Path(args.events),
            time_steps=args.steps,
            ion_count=args.ions,
            seed=args.seed,
        )
    )
    print(f"run_status={result.run_status}")
    if result.reason:
        print(f"reason={result.reason}")
    print(f"manifest={result.manifest_path}")
    print(f"profile_timeline_written={str(result.timeline_path.exists()).lower()}")
    print(f"transport_field_written={str(result.transport_field_path.exists()).lower()}")
    print(f"click_index_written={str(result.click_index_path.exists()).lower()}")
    print(f"uncertainty_map_written={str(result.uncertainty_map_path.exists()).lower()}")
    print(f"active_learning_plan_written={str(result.active_learning_plan_path.exists()).lower()}")
    print(f"surrogate_model_written={str((Path(args.out) / 'empirical_mdn_model.json').exists()).lower()}")
    print(f"surrogate_gate_written={str((Path(args.out) / 'surrogate_training_gate_report.json').exists()).lower()}")
    print(f"surrogate_registry_written={str((Path(args.out) / 'surrogate_model_manifest.json').exists()).lower()}")
    print(f"qa_report_written={str(result.qa_report_path.exists()).lower()}")
    return 0 if result.run_status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
