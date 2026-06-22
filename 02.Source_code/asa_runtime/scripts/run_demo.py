from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.runner import OfflineRunRequest, RunMode, run_offline_simulation


DemoName = Literal["pr_hole_3d", "pr_trench_2d"]


@dataclass(frozen=True, slots=True)
class DemoSpec:
    name: DemoName
    run_id: str
    mode: RunMode
    scene_path: Path | None
    image_path: Path | None
    kernel_path: Path
    events_path: Path
    ion_count: int


class DemoCliError(ValueError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=("pr_hole_3d", "pr_trench_2d"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ions", type=int)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--flux-ions-cm2-s", type=float, default=1.0e15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    demo_name = _demo_name(args.demo)
    spec = _demo_spec(demo_name, args.ions)
    result = run_offline_simulation(
        OfflineRunRequest(
            run_id=spec.run_id,
            mode=spec.mode,
            source_root=SOURCE_ROOT,
            output_dir=Path(args.out),
            scene_path=spec.scene_path,
            image_path=spec.image_path,
            kernel_path=spec.kernel_path,
            events_path=spec.events_path,
            time_steps=args.steps,
            ion_count=spec.ion_count,
            seed=args.seed,
            process_duration_s=args.duration_s,
            flux_ions_cm2_s=args.flux_ions_cm2_s,
        )
    )
    demo_complete = result.run_status == "complete"
    print(f"demo={spec.name}")
    print(f"demo_complete={str(demo_complete).lower()}")
    print(f"run_status={result.run_status}")
    if result.reason:
        print(f"reason={result.reason}")
    print(f"manifest={result.manifest_path}")
    print(f"profile_timeline_written={str(result.timeline_path.exists()).lower()}")
    print(f"transport_field_written={str(result.transport_field_path.exists()).lower()}")
    print(f"hit_history_written={str(result.hit_history_path.exists()).lower()}")
    print(f"click_index_written={str(result.click_index_path.exists()).lower()}")
    print(f"surrogate_model_written={str((Path(args.out) / 'empirical_mdn_model.json').exists()).lower()}")
    print(f"surrogate_gate_written={str((Path(args.out) / 'surrogate_training_gate_report.json').exists()).lower()}")
    print(f"surrogate_registry_written={str((Path(args.out) / 'surrogate_model_manifest.json').exists()).lower()}")
    print(f"qa_report_written={str(result.qa_report_path.exists()).lower()}")
    print(f"process_duration_s={args.duration_s}")
    print(f"flux_ions_cm2_s={args.flux_ions_cm2_s}")
    return 0 if demo_complete else 1


def _demo_name(raw: str) -> DemoName:
    match raw:
        case "pr_hole_3d":
            return "pr_hole_3d"
        case "pr_trench_2d":
            return "pr_trench_2d"
        case _:
            raise DemoCliError(f"unknown_demo={raw}")


def _demo_spec(name: DemoName, requested_ion_count: int | None) -> DemoSpec:
    fixture_root = SOURCE_ROOT / "tests" / "fixtures"
    ion_count = requested_ion_count or 8
    match name:
        case "pr_hole_3d":
            return DemoSpec(
                name=name,
                run_id="demo-pr-hole-3d",
                mode="3d",
                scene_path=fixture_root / "scenes" / "pr_hole_scene.json",
                image_path=None,
                kernel_path=fixture_root / "kernels" / "offline_ar_si_kernel.json",
                events_path=fixture_root / "md_events" / "md_events_small.jsonl",
                ion_count=ion_count,
            )
        case "pr_trench_2d":
            return DemoSpec(
                name=name,
                run_id="demo-pr-trench-2d",
                mode="2d",
                scene_path=None,
                image_path=fixture_root / "geometry" / "pr_trench.png",
                kernel_path=fixture_root / "kernels" / "offline_ar_si_kernel.json",
                events_path=fixture_root / "md_events" / "md_events_small.jsonl",
                ion_count=ion_count,
            )
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    raise SystemExit(main())
