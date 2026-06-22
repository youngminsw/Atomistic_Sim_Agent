from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_float, as_mapping, require
from sim_agent.schemas.errors import SchemaValidationError


MIN_MD_ATOM_COUNT: Final = 3_000
MAX_MD_ATOM_COUNT: Final = 7_000
MIN_FIXED_DEPTH_NM: Final = 1.0
MIN_THERMOSTAT_DEPTH_NM: Final = 1.0
MIN_RUN_LENGTH_PS: Final = 1.0
MAX_TIMESTEP_FS: Final = 0.25
MIN_LATERAL_SIZE_NM: Final = 5.0
CASCADE_LATERAL_FACTOR: Final = 2.0
CASCADE_MOBILE_DEPTH_FACTOR: Final = 1.5


@dataclass(frozen=True, slots=True)
class MDBoxReadinessReport:
    payload: JsonMap
    evidence: tuple[str, ...]
    blockers: tuple[str, ...]


def assess_md_box_readiness(md_box_value: object) -> MDBoxReadinessReport:
    blockers: list[str] = []
    evidence: list[str] = []
    if md_box_value is None:
        return MDBoxReadinessReport(
            payload={},
            evidence=(),
            blockers=("md_box_metadata_missing",),
        )
    evidence.append("md_box_metadata_present")
    try:
        md_box = as_mapping(md_box_value, "md_box")
    except SchemaValidationError:
        return MDBoxReadinessReport(
            payload={},
            evidence=tuple(evidence),
            blockers=("md_box_metadata_invalid",),
        )

    _record_md_box(md_box, blockers, evidence)
    return MDBoxReadinessReport(
        payload=md_box,
        evidence=tuple(evidence),
        blockers=tuple(blockers),
    )


def _record_md_box(md_box: JsonMap, blockers: list[str], evidence: list[str]) -> None:
    blocker_count = len(blockers)
    x_nm = _positive_float(md_box, "x_nm", blockers)
    y_nm = _positive_float(md_box, "y_nm", blockers)
    mobile_depth_nm = _positive_float(md_box, "mobile_depth_nm", blockers)
    fixed_depth_nm = _positive_float(md_box, "fixed_depth_nm", blockers)
    thermostat_depth_nm = _positive_float(md_box, "thermostat_depth_nm", blockers)
    cascade_depth_nm = _positive_float(md_box, "expected_cascade_depth_nm", blockers)
    timestep_fs = _positive_float(md_box, "timestep_fs", blockers)
    run_length_ps = _positive_float(md_box, "run_length_ps", blockers)
    atom_count = _positive_int_value(md_box, "atom_count", blockers)
    if len(blockers) != blocker_count:
        return

    required_lateral_nm = max(MIN_LATERAL_SIZE_NM, cascade_depth_nm * CASCADE_LATERAL_FACTOR)
    required_mobile_depth_nm = cascade_depth_nm * CASCADE_MOBILE_DEPTH_FACTOR
    _record_min_float("md_box_lateral_size_too_small:x_nm", x_nm, required_lateral_nm, blockers)
    _record_min_float("md_box_lateral_size_too_small:y_nm", y_nm, required_lateral_nm, blockers)
    _record_min_float(
        "md_box_mobile_depth_too_shallow",
        mobile_depth_nm,
        required_mobile_depth_nm,
        blockers,
    )
    _record_min_float("md_box_fixed_region_too_thin", fixed_depth_nm, MIN_FIXED_DEPTH_NM, blockers)
    _record_min_float(
        "md_box_thermostat_region_too_thin",
        thermostat_depth_nm,
        MIN_THERMOSTAT_DEPTH_NM,
        blockers,
    )
    _record_max_float("md_box_timestep_too_large", timestep_fs, MAX_TIMESTEP_FS, blockers)
    _record_min_float("md_box_run_length_too_short", run_length_ps, MIN_RUN_LENGTH_PS, blockers)
    if atom_count < MIN_MD_ATOM_COUNT:
        blockers.append(f"md_box_atom_count_too_low:{atom_count}<{MIN_MD_ATOM_COUNT}")
    if atom_count > MAX_MD_ATOM_COUNT:
        blockers.append(f"md_box_atom_count_too_high:{atom_count}>{MAX_MD_ATOM_COUNT}")
    if len(blockers) == blocker_count:
        evidence.extend(
            (
                "md_box_size_sufficient",
                "md_box_regions_sufficient",
                "md_box_timestep_run_length_sufficient",
                "md_box_atom_count_sufficient",
            )
        )


def _positive_float(payload: JsonMap, field: str, blockers: list[str]) -> float:
    try:
        value = as_float(require(payload, field), field)
    except SchemaValidationError:
        blockers.append(f"md_box_{field}_invalid")
        return 0.0
    if value <= 0.0:
        blockers.append(f"md_box_{field}_invalid")
        return 0.0
    return value


def _positive_int_value(payload: JsonMap, field: str, blockers: list[str]) -> int:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    blockers.append(f"md_box_{field}_invalid")
    return 0


def _record_min_float(
    code: str,
    value: float,
    minimum: float,
    blockers: list[str],
) -> None:
    if value < minimum:
        if ":" in code:
            prefix, field = code.split(":", maxsplit=1)
            blockers.append(f"{prefix}:{field}={value:.1f}<{minimum:.1f}")
            return
        blockers.append(f"{code}:{value:.1f}<{minimum:.1f}")


def _record_max_float(
    code: str,
    value: float,
    maximum: float,
    blockers: list[str],
) -> None:
    if value > maximum:
        blockers.append(f"{code}:{value:.2f}>{maximum:.2f}")
