from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sim_agent.schemas.distributions import IonAngularDistribution, IonEnergyDistribution


RunMode = Literal["2d", "3d"]


class RunManagerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OfflineRunRequest:
    run_id: str
    mode: RunMode
    source_root: Path
    output_dir: Path
    scene_path: Path | None
    image_path: Path | None
    kernel_path: Path
    events_path: Path
    time_steps: int
    ion_count: int
    seed: int
    time_step_s: float = 0.1
    cell_area_nm2: float = 2.0
    pixel_size_nm: float = 1.0
    process_duration_s: float = 600.0
    flux_ions_cm2_s: float = 1.0e15
    energy_distribution: IonEnergyDistribution | None = None
    angular_distribution: IonAngularDistribution | None = None


@dataclass(frozen=True, slots=True)
class OfflineRunResult:
    run_id: str
    run_status: str
    output_dir: Path
    manifest_path: Path
    timeline_path: Path
    transport_field_path: Path
    hit_history_path: Path
    click_index_path: Path
    uncertainty_map_path: Path
    active_learning_plan_path: Path
    qa_report_path: Path
    artifact_count: int
    reason: str = ""
