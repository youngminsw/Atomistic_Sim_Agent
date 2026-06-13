from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.md import MDRunStatus, MDVerificationReport, verify_md_run
from sim_agent.ml_surrogate import (
    CoverageRange,
    EMPIRICAL_MDN_ARTIFACT,
    EMPIRICAL_MDN_BACKEND,
    KernelCoverage,
    KernelFeatureSpec,
    MDNTrainingMetrics,
    SurrogateDatasetError,
    SurrogateTrainingCriteria,
    SurrogateTrainingDataset,
    assess_surrogate_training_readiness,
    build_training_dataset,
    register_surrogate_model,
    surrogate_training_gate_report_payload,
    write_empirical_mdn_model,
)
from sim_agent.ml_surrogate.coverage import coverage_from_dataset
from sim_agent.ml_surrogate.kernel import (
    DEFAULT_FORCE_FIELD_PROTOCOL_ID,
    DEFAULT_PHYSICS_SCOPE,
    InteractionKernelManifest,
)
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
    parser.add_argument("--validation-event-count", type=int, required=True)
    parser.add_argument("--validation-nll", type=float, required=True)
    parser.add_argument("--deposited-energy-mae-eV", type=float, required=True)
    parser.add_argument("--sputter-yield-mae", type=float, required=True)
    parser.add_argument("--reflection-brier-score", type=float, required=True)
    parser.add_argument("--calibration-error", type=float, required=True)
    parser.add_argument("--high-uncertainty-fraction", type=float, required=True)
    parser.add_argument("--min-training-events", type=int, default=1000)
    parser.add_argument("--min-validation-events", type=int, default=200)
    parser.add_argument("--max-validation-nll", type=float, default=0.25)
    parser.add_argument("--max-deposited-energy-mae-eV", type=float, default=5.0)
    parser.add_argument("--max-sputter-yield-mae", type=float, default=0.2)
    parser.add_argument("--max-reflection-brier-score", type=float, default=0.1)
    parser.add_argument("--max-calibration-error", type=float, default=0.08)
    parser.add_argument("--max-high-uncertainty-fraction", type=float, default=0.01)
    parser.add_argument("--required-energy-range-eV", required=True)
    parser.add_argument("--required-polar-range-deg", required=True)
    parser.add_argument("--required-azimuth-range-deg", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    try:
        return _run(args)
    except (
        json.JSONDecodeError,
        OSError,
        SchemaValidationError,
        SurrogateDatasetError,
        ValueError,
    ) as exc:
        print("mdn_training_ok=false")
        print("surrogate_training_ok=false")
        print(str(exc))
        return 1


def _run(args: argparse.Namespace) -> int:
    output_dir = Path(args.out_dir)
    spec = KernelFeatureSpec.from_mapping(
        as_mapping(json.loads(Path(args.kernel).read_text(encoding="utf-8")), "kernel")
    )
    report = verify_md_run(
        log_path=Path(args.log),
        events_path=Path(args.events),
        expected_events=args.expected_events,
        required_ion=args.required_ion,
        required_material=args.required_material,
    )
    dataset = build_training_dataset(_verified_report(report), spec)
    manifest = _manifest(spec, dataset, args.events)
    metrics = _metrics(args)
    gate = assess_surrogate_training_readiness(manifest, metrics, _criteria(args))
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "surrogate_training_gate_report.json"
    gate_path.write_text(
        json.dumps(surrogate_training_gate_report_payload(gate), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = write_empirical_mdn_model(output_dir, manifest, dataset)

    registry_path = ""
    if gate.accepted:
        registration = register_surrogate_model(
            output_dir,
            manifest,
            metrics,
            gate,
            EMPIRICAL_MDN_ARTIFACT,
        )
        registry_path = str(registration.registry_path)

    print(f"mdn_training_ok={str(gate.accepted).lower()}")
    print(f"surrogate_training_ok={str(gate.accepted).lower()}")
    print(f"training_backend={EMPIRICAL_MDN_BACKEND}")
    print(f"quality_gate_decision={gate.decision}")
    print(f"blockers={','.join(gate.blockers)}")
    print(f"next_actions={','.join(gate.next_actions)}")
    print(f"model_artifact={model.artifact_path}")
    print(f"gate_report={gate_path}")
    print(f"registered_for_feature_scale={str(gate.accepted).lower()}")
    if registry_path:
        print(f"registry_path={registry_path}")
    return 0 if gate.accepted else 1


def _verified_report(report: MDVerificationReport) -> MDVerificationReport:
    if report.status is not MDRunStatus.VERIFIED or not report.ok:
        raise SurrogateDatasetError("verified_md_required")
    return report


def _manifest(
    spec: KernelFeatureSpec,
    dataset: SurrogateTrainingDataset,
    provenance_source: str,
) -> InteractionKernelManifest:
    return InteractionKernelManifest(
        kernel_id=dataset.kernel_id,
        ion_species=spec.ion_species,
        material_id=spec.material_id,
        force_field_protocol_id=DEFAULT_FORCE_FIELD_PROTOCOL_ID,
        physics_scope=DEFAULT_PHYSICS_SCOPE,
        training_event_count=dataset.row_count,
        coverage=coverage_from_dataset(dataset),
        provenance_sources=(provenance_source,),
    )


def _metrics(args: argparse.Namespace) -> MDNTrainingMetrics:
    return MDNTrainingMetrics(
        validation_event_count=args.validation_event_count,
        validation_nll=args.validation_nll,
        deposited_energy_mae_eV=args.deposited_energy_mae_eV,
        sputter_yield_mae=args.sputter_yield_mae,
        reflection_brier_score=args.reflection_brier_score,
        calibration_error=args.calibration_error,
        high_uncertainty_fraction=args.high_uncertainty_fraction,
    )


def _criteria(args: argparse.Namespace) -> SurrogateTrainingCriteria:
    return SurrogateTrainingCriteria(
        min_training_events=args.min_training_events,
        min_validation_events=args.min_validation_events,
        max_validation_nll=args.max_validation_nll,
        max_deposited_energy_mae_eV=args.max_deposited_energy_mae_eV,
        max_sputter_yield_mae=args.max_sputter_yield_mae,
        max_reflection_brier_score=args.max_reflection_brier_score,
        max_calibration_error=args.max_calibration_error,
        max_high_uncertainty_fraction=args.max_high_uncertainty_fraction,
        required_coverage=KernelCoverage(
            energy_eV=_range(args.required_energy_range_eV),
            polar_deg=_range(args.required_polar_range_deg),
            azimuth_deg=_range(args.required_azimuth_range_deg),
        ),
    )


def _range(raw: str) -> CoverageRange:
    parts = raw.split(":", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("range_must_be_min_colon_max")
    minimum = float(parts[0])
    maximum = float(parts[1])
    if maximum < minimum:
        raise ValueError("range_max_must_be_at_least_min")
    return CoverageRange(minimum=minimum, maximum=maximum)


if __name__ == "__main__":
    raise SystemExit(main())
