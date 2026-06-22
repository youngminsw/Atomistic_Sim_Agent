from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap

from .types import TransportError


NM2_TO_CM2 = 1.0e-14


@dataclass(frozen=True, slots=True)
class FeatureScaleProcessSchedule:
    duration_s: float
    flux_ions_cm2_s: float
    active_area_nm2: float
    sampled_ion_count: int

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise TransportError("process_duration_must_be_positive")
        if self.flux_ions_cm2_s <= 0.0:
            raise TransportError("flux_must_be_positive")
        if self.active_area_nm2 <= 0.0:
            raise TransportError("active_area_must_be_positive")
        if self.sampled_ion_count <= 0:
            raise TransportError("sampled_ion_count_must_be_positive")

    @property
    def fluence_ions_cm2(self) -> float:
        return self.flux_ions_cm2_s * self.duration_s

    @property
    def active_area_cm2(self) -> float:
        return self.active_area_nm2 * NM2_TO_CM2

    @property
    def physical_incident_count(self) -> float:
        return self.fluence_ions_cm2 * self.active_area_cm2

    @property
    def regular_ion_interval_s(self) -> float:
        return self.duration_s / self.sampled_ion_count

    @property
    def sample_weight(self) -> float:
        return self.physical_incident_count / self.sampled_ion_count


def process_schedule_payload(schedule: FeatureScaleProcessSchedule) -> JsonMap:
    return {
        "duration_s": schedule.duration_s,
        "duration_min": schedule.duration_s / 60.0,
        "flux_ions_cm2_s": schedule.flux_ions_cm2_s,
        "fluence_ions_cm2": schedule.fluence_ions_cm2,
        "active_area_nm2": schedule.active_area_nm2,
        "active_area_cm2": schedule.active_area_cm2,
        "physical_incident_count": schedule.physical_incident_count,
        "sampled_ion_count": schedule.sampled_ion_count,
        "regular_ion_interval_s": schedule.regular_ion_interval_s,
        "sample_weight": schedule.sample_weight,
        "sampling_policy": "regular_time_interval_weighted_ions",
    }
