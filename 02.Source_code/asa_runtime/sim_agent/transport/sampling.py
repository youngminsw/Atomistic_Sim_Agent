from __future__ import annotations

from random import Random

from sim_agent.schemas.distributions import IonAngularDistribution, IonEnergyBin, IonEnergyDistribution

from .types import IonSample, TransportError


def sample_ions(
    energy_distribution: IonEnergyDistribution,
    angular_distribution: IonAngularDistribution,
    ion_count: int,
    seed: int,
    duration_s: float | None = None,
) -> tuple[IonSample, ...]:
    if ion_count <= 0:
        raise TransportError("ion_count_must_be_positive")
    interval_s = _regular_interval_s(ion_count, duration_s)
    rng = Random(seed)
    return tuple(
        IonSample(
            event_id=f"transport-ion-{index:06d}",
            energy_eV=_sample_energy(energy_distribution.bins, rng),
            polar_deg=rng.uniform(angular_distribution.polar_min_deg, angular_distribution.polar_max_deg),
            azimuth_deg=rng.uniform(angular_distribution.azimuth_min_deg, angular_distribution.azimuth_max_deg),
            time_step=index,
            time_s=round(index * interval_s, 12),
        )
        for index in range(ion_count)
    )


def _regular_interval_s(ion_count: int, duration_s: float | None) -> float:
    if duration_s is None:
        return 1.0
    if duration_s <= 0.0:
        raise TransportError("process_duration_must_be_positive")
    return duration_s / ion_count


def _sample_energy(bins: tuple[IonEnergyBin, ...], rng: Random) -> float:
    if not bins:
        raise TransportError("iedf_bins_required")
    total_probability = sum(item.probability for item in bins)
    if total_probability <= 0.0:
        raise TransportError("iedf_probability_must_be_positive")
    cursor = rng.uniform(0.0, total_probability)
    cumulative = 0.0
    for item in bins:
        cumulative += item.probability
        if cursor <= cumulative:
            return rng.uniform(item.min, item.max)
    last = bins[-1]
    return rng.uniform(last.min, last.max)
