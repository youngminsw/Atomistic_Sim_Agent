from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim_agent.schemas.common import UncertaintyReport

from .types import SurrogateTrainingDataset

if TYPE_CHECKING:
    from .kernel import InteractionContext


@dataclass(frozen=True, slots=True)
class CoverageRange:
    minimum: float
    maximum: float

    def gap(self, value: float) -> float:
        if value < self.minimum:
            return self.minimum - value
        if value > self.maximum:
            return value - self.maximum
        return 0.0

    @property
    def scale(self) -> float:
        return max(self.maximum - self.minimum, 1.0)


@dataclass(frozen=True, slots=True)
class KernelCoverage:
    energy_eV: CoverageRange
    polar_deg: CoverageRange
    azimuth_deg: CoverageRange


def coverage_from_dataset(dataset: SurrogateTrainingDataset) -> KernelCoverage:
    energy_values = tuple(row.feature_vector[0] for row in dataset.rows)
    polar_values = tuple(row.feature_vector[1] for row in dataset.rows)
    azimuth_values = tuple(row.feature_vector[2] for row in dataset.rows)
    return KernelCoverage(
        energy_eV=CoverageRange(min(energy_values), max(energy_values)),
        polar_deg=CoverageRange(min(polar_values), max(polar_values)),
        azimuth_deg=CoverageRange(min(azimuth_values), max(azimuth_values)),
    )


def uncertainty_for_context(coverage: KernelCoverage, context: InteractionContext) -> UncertaintyReport:
    score = min(
        1.0,
        coverage.energy_eV.gap(context.energy_eV) / coverage.energy_eV.scale
        + coverage.polar_deg.gap(context.polar_deg) / coverage.polar_deg.scale
        + coverage.azimuth_deg.gap(context.azimuth_deg) / coverage.azimuth_deg.scale,
    )
    return UncertaintyReport(
        score=score if score > 0.0 else 0.1,
        ood=score > 0.0,
        reason="coverage_gap" if score > 0.0 else None,
    )
