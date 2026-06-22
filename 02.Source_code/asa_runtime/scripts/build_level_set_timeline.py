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
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
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
    parser.add_argument("--click-nm", required=True)
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
        click_x, click_y, click_z = _triple_float(args.click_nm)
        diagnostic = timeline.diagnostic_at_nm(geometry, click_x, click_y, click_z)
    except (
        json.JSONDecodeError,
        OSError,
        GeometryError,
        KMCTransportError,
        LevelSetError,
        SchemaValidationError,
        SurrogateDatasetError,
    ) as exc:
        print("level_set_ok=false")
        print(str(exc))
        return 1

    print("level_set_ok=true")
    print(f"feature_type={timeline.feature_type.value}")
    print(f"state_count={timeline.state_count}")
    print(f"final_removed_volume_nm3={timeline.final_state.total_removed_volume_nm3:.3f}")
    print(f"click_material={diagnostic.material_id}")
    print(f"click_region={diagnostic.region}")
    print(f"click_depth_history_nm={_format_series(diagnostic.depth_history_nm)}")
    print(f"click_energy_history_eV={_format_series(diagnostic.energy_history_eV)}")
    return 0


def _impact(raw: str) -> IonImpact:
    event_id, coordinates = _split_once(raw, ":")
    x_nm, y_nm, z_nm = _triple_float(coordinates)
    return IonImpact(event_id=event_id, x_nm=x_nm, y_nm=y_nm, z_nm=z_nm, time_step=0)


def _split_once(raw: str, delimiter: str) -> tuple[str, str]:
    parts = raw.split(delimiter, maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise LevelSetError("invalid_delimited_value")
    return (parts[0], parts[1])


def _triple_float(raw: str) -> tuple[float, float, float]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise LevelSetError("expected_three_coordinates")
    return (_float(parts[0]), _float(parts[1]), _float(parts[2]))


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise LevelSetError(f"invalid_float={raw}") from exc


def _format_series(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.3f}" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
