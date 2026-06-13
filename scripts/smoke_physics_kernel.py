from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.ml_surrogate import (
    InteractionContext,
    InteractionKernelError,
    KernelFeatureSpec,
    SurrogateDatasetError,
    UncertaintyMapSample,
    build_fixture_interaction_kernel,
    write_uncertainty_map,
)
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


DEFAULT_KERNEL = SOURCE_ROOT / "tests" / "fixtures" / "kernels" / "offline_ar_si_kernel.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    parser.add_argument("--azimuth-deg", type=float, default=120.0)
    parser.add_argument("--uncertainty-map-out")
    parser.add_argument("--run-id", default="smoke-physics-kernel")
    parser.add_argument("--sample-id", default="smoke-sample-001")
    parser.add_argument("--snapshot-id", default="smoke-snapshot-001")
    args = parser.parse_args()

    try:
        spec = KernelFeatureSpec.from_mapping(
            as_mapping(json.loads(Path(args.kernel).read_text(encoding="utf-8")), "kernel")
        )
        kernel = build_fixture_interaction_kernel(Path(args.fixture), spec, provenance_source=args.fixture)
        context = _context(args.context, args.azimuth_deg)
        inference = kernel.registry().infer(context)
    except (
        json.JSONDecodeError,
        OSError,
        SchemaValidationError,
        SurrogateDatasetError,
        InteractionKernelError,
        ValueError,
    ) as exc:
        print("event_bundle_valid=false")
        print(str(exc))
        return 1

    bundle = inference.bundle
    print("event_bundle_valid=true")
    print(f"kernel_id={kernel.manifest.kernel_id}")
    print(f"reflection_probability={bundle.reflection.probability}")
    print(f"sputtering_yield={bundle.sputtering.yield_atoms_per_ion}")
    print(f"energy_transfer_eV={bundle.energy_transfer.deposited_energy_eV:.1f}")
    print(f"removed_depth_nm={bundle.removed_depth_nm:.3f}")
    print(f"uncertainty_score={bundle.uncertainty.score:.3f}")
    print(f"ood={str(bundle.uncertainty.ood).lower()}")
    print(f"active_learning_suggested={str(inference.active_learning_suggested).lower()}")
    if args.uncertainty_map_out is not None:
        write_uncertainty_map(
            Path(args.uncertainty_map_out),
            args.run_id,
            kernel.manifest,
            (UncertaintyMapSample(args.sample_id, context, inference, args.snapshot_id),),
        )
        print(f"uncertainty_map={args.uncertainty_map_out}")
    return 0


def _context(raw: str, azimuth_deg: float) -> InteractionContext:
    parts = raw.lower().split("_")
    if len(parts) != 4:
        raise InteractionKernelError("invalid_context")
    energy = _number_with_suffix(parts[2], "ev")
    polar = _number_with_suffix(parts[3], "deg")
    return InteractionContext(
        ion_species=parts[0].capitalize(),
        material_id=parts[1].capitalize(),
        force_field_protocol_id="Si_Tersoff_ZBL_physical_v001",
        physics_scope="physical_bombardment_no_chemistry",
        energy_eV=energy,
        polar_deg=polar,
        azimuth_deg=azimuth_deg,
        local_incidence_deg=polar,
        phase="crystal",
        amorphous_index=0.0,
        roughness_rms_nm=0.1,
        rdf_crystal_similarity=0.92,
        rdf_amorphous_similarity=0.08,
        damage_dose=0.0,
        implanted_inert_fraction=0.0,
        local_fluence=0.0,
        removed_depth_nm=0.0,
    )


def _number_with_suffix(raw: str, suffix: str) -> float:
    if not raw.endswith(suffix):
        raise InteractionKernelError("invalid_context")
    return float(raw[: -len(suffix)])


if __name__ == "__main__":
    raise SystemExit(main())
