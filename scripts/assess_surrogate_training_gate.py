from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.ml_surrogate import (
    CoverageRange,
    DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION,
    KernelCoverage,
    KernelFeatureSpec,
    MDNTrainingMetrics,
    SurrogateDatasetError,
    SurrogateTrainingCriteria,
    assess_surrogate_training_readiness,
    build_fixture_interaction_kernel,
    surrogate_training_gate_report_payload,
)
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--kernel", required=True)
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
    parser.add_argument(
        "--max-high-uncertainty-fraction",
        type=float,
        default=DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION,
    )
    parser.add_argument("--required-energy-range-eV", required=True)
    parser.add_argument("--required-polar-range-deg", required=True)
    parser.add_argument("--required-azimuth-range-deg", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        spec = KernelFeatureSpec.from_mapping(
            as_mapping(json.loads(Path(args.kernel).read_text(encoding="utf-8")), "kernel")
        )
        kernel = build_fixture_interaction_kernel(Path(args.events), spec, provenance_source=args.events)
        report = assess_surrogate_training_readiness(
            kernel.manifest,
            _metrics(args),
            _criteria(args),
        )
    except (
        json.JSONDecodeError,
        OSError,
        SchemaValidationError,
        SurrogateDatasetError,
        ValueError,
    ) as exc:
        print("surrogate_training_gate_ok=false")
        print(str(exc))
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(surrogate_training_gate_report_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"surrogate_training_gate_ok={str(report.accepted).lower()}")
    print(f"decision={report.decision}")
    print(f"blockers={','.join(report.blockers)}")
    print(f"next_actions={','.join(report.next_actions)}")
    print(f"report_path={out_path}")
    return 0 if report.accepted else 1


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
