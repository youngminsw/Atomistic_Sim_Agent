from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.geometry import GeometryError, GridShape, load_pattern_geometry_from_scene
from sim_agent.ml_surrogate import (
    InteractionKernelError,
    KernelFeatureSpec,
    SurrogateDatasetError,
    build_fixture_interaction_kernel,
)
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.distributions import IonAngularDistribution, IonEnergyBin, IonEnergyDistribution
from sim_agent.schemas.errors import SchemaValidationError
from sim_agent.transport import TransportError, TransportResult, run_transport_3d


DEFAULT_OUT = SOURCE_ROOT / "evidence" / "smoke-transport-field.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--ions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    kernel_path = Path(args.kernel)
    if not kernel_path.exists():
        print("transport_valid=false")
        print("kernel_not_found")
        return 1

    try:
        scene = as_mapping(json.loads(Path(args.scene).read_text(encoding="utf-8")), "scene")
        geometry = load_pattern_geometry_from_scene(scene, SOURCE_ROOT, GridShape(32, 32, 16), target_depth_nm=24.0)
        manifest_before = geometry.export_manifest()
        spec = KernelFeatureSpec.from_mapping(as_mapping(json.loads(kernel_path.read_text(encoding="utf-8")), "kernel"))
        registry = build_fixture_interaction_kernel(Path(args.events), spec, provenance_source=args.events).registry()
        result = run_transport_3d(
            geometry=geometry,
            registry=registry,
            energy_distribution=_energy_distribution(),
            angular_distribution=_angular_distribution(),
            ion_count=args.ions,
            seed=args.seed,
        )
        out_path = Path(args.out)
        _write_field_summary(out_path, result)
    except (
        json.JSONDecodeError,
        OSError,
        GeometryError,
        InteractionKernelError,
        SchemaValidationError,
        SurrogateDatasetError,
        TransportError,
    ) as exc:
        print("transport_valid=false")
        print(str(exc))
        return 1

    print("transport_valid=true")
    print(f"feature_type={result.feature_type}")
    print("energy_field_written=true")
    print(f"field_path={out_path}")
    print(f"hit_history_count={result.hit_history_count}")
    print(f"cell_count={result.field.cell_count}")
    print(f"total_deposited_energy_eV={result.field.total_deposited_energy_eV:.1f}")
    print(f"geometry_mutated={str(geometry.export_manifest() != manifest_before).lower()}")
    return 0


def _energy_distribution() -> IonEnergyDistribution:
    return IonEnergyDistribution(
        kind="histogram",
        unit="eV",
        bins=(
            IonEnergyBin(min=80.0, max=90.0, probability=0.5),
            IonEnergyBin(min=90.0, max=100.0, probability=0.5),
        ),
    )


def _angular_distribution() -> IonAngularDistribution:
    return IonAngularDistribution(
        kind="uniform",
        polar_min_deg=30.0,
        polar_max_deg=45.0,
        azimuth_min_deg=120.0,
        azimuth_max_deg=240.0,
    )


def _write_field_summary(path: Path, result: TransportResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": result.mode,
        "feature_type": result.feature_type,
        "hit_history_count": result.hit_history_count,
        "cell_count": result.field.cell_count,
        "total_deposited_energy_eV": result.field.total_deposited_energy_eV,
        "total_removed_depth_nm": result.field.total_removed_depth_nm,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
