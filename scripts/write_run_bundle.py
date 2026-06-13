from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.geometry import GeometryError, GridShape, load_pattern_geometry_from_scene
from sim_agent.kmc import IonImpact, KMCTransportError, accumulate_energy_deposition
from sim_agent.level_set import LevelSetConfig, LevelSetError, evolve_profile
from sim_agent.md import verify_md_run
from sim_agent.ml_surrogate import KernelFeatureSpec, SurrogateDatasetError, build_training_dataset
from sim_agent.run_artifacts import RunArtifactError, write_profile_run_bundle
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    try:
        scene = as_mapping(json.loads(Path(args.scene).read_text(encoding="utf-8")), "scene")
        geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
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
        field = accumulate_energy_deposition(geometry, dataset, tuple(_impact(item) for item in args.impact))
        timeline = evolve_profile(
            field,
            LevelSetConfig(
                time_steps=args.time_steps,
                time_step_s=args.time_step_s,
                cell_area_nm2=args.cell_area_nm2,
            ),
        )
        bundle = write_profile_run_bundle(
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
            geometry=geometry,
            timeline=timeline,
            click_points_nm=tuple(_triple_float(item) for item in args.click_nm),
        )
    except (
        json.JSONDecodeError,
        OSError,
        GeometryError,
        KMCTransportError,
        LevelSetError,
        RunArtifactError,
        SchemaValidationError,
        SurrogateDatasetError,
    ) as exc:
        print("run_bundle_ok=false")
        print(str(exc))
        return 1

    print("run_bundle_ok=true")
    print(f"run_id={bundle.run_id}")
    print(f"artifact_count={bundle.artifact_count}")
    print(f"manifest={bundle.manifest_path}")
    print(f"timeline={bundle.timeline_path}")
    print(f"diagnostics={bundle.diagnostics_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--required-ion")
    parser.add_argument("--required-material")
    parser.add_argument("--impact", action="append", required=True)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--time-step-s", type=float, required=True)
    parser.add_argument("--cell-area-nm2", type=float, required=True)
    parser.add_argument("--click-nm", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _impact(raw: str) -> IonImpact:
    event_id, coordinates = _split_once(raw, ":")
    x_nm, y_nm, z_nm = _triple_float(coordinates)
    return IonImpact(event_id=event_id, x_nm=x_nm, y_nm=y_nm, z_nm=z_nm, time_step=0)


def _split_once(raw: str, delimiter: str) -> tuple[str, str]:
    parts = raw.split(delimiter, maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise RunArtifactError("invalid_delimited_value")
    return (parts[0], parts[1])


def _triple_float(raw: str) -> tuple[float, float, float]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise RunArtifactError("expected_three_coordinates")
    return (_float(parts[0]), _float(parts[1]), _float(parts[2]))


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise RunArtifactError(f"invalid_float={raw}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
