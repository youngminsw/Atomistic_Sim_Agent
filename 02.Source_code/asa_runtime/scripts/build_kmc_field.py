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
    args = parser.parse_args()

    try:
        scene = as_mapping(json.loads(Path(args.scene).read_text(encoding="utf-8")), "scene")
        geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
        manifest_before = geometry.export_manifest()
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
    except (
        json.JSONDecodeError,
        OSError,
        GeometryError,
        KMCTransportError,
        SchemaValidationError,
        SurrogateDatasetError,
    ) as exc:
        print("kmc_field_ok=false")
        print(str(exc))
        return 1

    print("kmc_field_ok=true")
    print(f"feature_type={field.feature_type.value}")
    print(f"hit_count={field.total_hit_count}")
    print(f"cell_count={field.cell_count}")
    print(f"total_deposited_energy_eV={field.total_deposited_energy_eV:.1f}")
    print(f"total_removal_drive_nm={field.total_removal_drive_nm:.3f}")
    print(f"geometry_mutated={str(geometry.export_manifest() != manifest_before).lower()}")
    return 0


def _impact(raw: str) -> IonImpact:
    event_id, coordinates = _split_once(raw, ":")
    x_raw, y_raw, z_raw = _triple(coordinates)
    return IonImpact(event_id=event_id, x_nm=_float(x_raw), y_nm=_float(y_raw), z_nm=_float(z_raw), time_step=0)


def _split_once(raw: str, delimiter: str) -> tuple[str, str]:
    parts = raw.split(delimiter, maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise KMCTransportError("invalid_impact_format")
    return (parts[0], parts[1])


def _triple(raw: str) -> tuple[str, str, str]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise KMCTransportError("impact_requires_three_coordinates")
    return (parts[0], parts[1], parts[2])


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise KMCTransportError(f"invalid_float={raw}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
