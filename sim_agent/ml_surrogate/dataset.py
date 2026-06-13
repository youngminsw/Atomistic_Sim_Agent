from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from sim_agent.schemas._parse import JsonMap
from sim_agent.md import MDVerificationReport, ParsedMDEvent

from .types import (
    KernelFeatureSpec,
    SurrogateDatasetError,
    SurrogateTargets,
    SurrogateTrainingDataset,
    SurrogateTrainingRow,
)


@dataclass(frozen=True, slots=True)
class SurrogateDatasetAuditReport:
    ok: bool
    payload: JsonMap


def build_training_dataset(report: MDVerificationReport, spec: KernelFeatureSpec) -> SurrogateTrainingDataset:
    if not report.ok or report.dataset is None:
        raise SurrogateDatasetError("verified_md_required")
    rows = tuple(_row(event, spec) for event in report.dataset.events)
    return SurrogateTrainingDataset(
        kernel_id=spec.kernel_id,
        feature_columns=spec.inputs,
        output_columns=spec.outputs,
        rows=rows,
    )


def audit_training_dataset(
    dataset: SurrogateTrainingDataset,
    min_events: int,
    required_outputs: tuple[str, ...],
) -> SurrogateDatasetAuditReport:
    blockers: list[str] = []
    evidence: list[str] = []
    if dataset.row_count >= min_events:
        evidence.append("dataset_event_count_sufficient")
    else:
        blockers.append(f"dataset_event_count_too_low:{dataset.row_count}<{min_events}")
    missing_outputs = tuple(output for output in required_outputs if output not in dataset.output_columns)
    if missing_outputs:
        blockers.append(f"dataset_outputs_missing:{','.join(missing_outputs)}")
    else:
        evidence.append("dataset_outputs_complete")
    _record_row_quality(dataset, blockers, evidence)
    ok = not blockers
    return SurrogateDatasetAuditReport(
        ok=ok,
        payload={
            "ok": ok,
            "kernel_id": dataset.kernel_id,
            "row_count": dataset.row_count,
            "feature_columns": dataset.feature_columns,
            "output_columns": dataset.output_columns,
            "total_removed_depth_nm": dataset.total_removed_depth_nm,
            "evidence": tuple(evidence),
            "blockers": tuple(blockers),
        },
    )


def _row(event: ParsedMDEvent, spec: KernelFeatureSpec) -> SurrogateTrainingRow:
    if event.ion != spec.ion_species:
        raise SurrogateDatasetError(f"kernel_ion_mismatch:{event.event_id}")
    if event.material_id != spec.material_id:
        raise SurrogateDatasetError(f"kernel_material_mismatch:{event.event_id}")
    return SurrogateTrainingRow(
        event_id=event.event_id,
        ion=event.ion,
        material_id=event.material_id,
        feature_vector=tuple(_features(event)[feature] for feature in spec.inputs),
        targets=SurrogateTargets(
            reflection_probability=1.0 if event.reflected else 0.0,
            sputter_probability=1.0 if event.event_type == "sputter" else 0.0,
            sputter_yield_atoms_per_ion=event.yield_atoms_per_ion,
            reflection_energy_out_eV=event.reflection_energy_out_eV,
            reflection_polar_deg=event.reflection_polar_deg,
            reflection_azimuth_deg=event.reflection_azimuth_deg,
            implant_retained_fraction=event.implant_retained_fraction,
            implant_depth_mean_nm=event.implant_depth_mean_nm,
            deposited_energy_eV=event.deposited_energy_eV,
            removed_depth_nm=event.removed_depth_nm,
        ),
    )


def _features(event: ParsedMDEvent) -> dict[str, float]:
    return {
        "energy_eV": event.energy_eV,
        "polar_deg": event.polar_deg,
        "azimuth_deg": event.azimuth_deg,
        "local_incidence_deg": event.polar_deg,
        "amorphous_index": event.amorphous_index,
        "roughness_rms_nm": event.roughness_rms_nm,
        "removed_depth_nm": event.prior_removed_depth_nm,
        "damage_dose": event.damage_dose,
        "implanted_inert_fraction": event.implanted_inert_fraction,
        "local_fluence": event.local_fluence,
        "rdf_crystal_similarity": event.rdf_crystal_similarity,
        "rdf_amorphous_similarity": event.rdf_amorphous_similarity,
    }


def _record_row_quality(
    dataset: SurrogateTrainingDataset,
    blockers: list[str],
    evidence: list[str],
) -> None:
    bad_rows: list[str] = []
    for row in dataset.rows:
        if not all(isfinite(value) for value in row.feature_vector):
            bad_rows.append(f"nonfinite_feature:{row.event_id}")
        target = row.targets
        if target.reflection_probability < 0.0 or target.reflection_probability > 1.0:
            bad_rows.append(f"reflection_probability_out_of_range:{row.event_id}")
        if target.sputter_probability < 0.0 or target.sputter_probability > 1.0:
            bad_rows.append(f"sputter_probability_out_of_range:{row.event_id}")
        if target.sputter_yield_atoms_per_ion < 0.0:
            bad_rows.append(f"sputter_yield_negative:{row.event_id}")
        if target.deposited_energy_eV < 0.0:
            bad_rows.append(f"deposited_energy_negative:{row.event_id}")
        if target.removed_depth_nm < 0.0:
            bad_rows.append(f"removed_depth_negative:{row.event_id}")
    if bad_rows:
        blockers.extend(bad_rows)
    else:
        evidence.append("dataset_rows_physically_sane")
