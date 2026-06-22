from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap


PRODUCTION_MIN_INCIDENTS: Final = 500


@dataclass(frozen=True, slots=True)
class MDProductionAcceptanceReport:
    accepted: bool
    payload: JsonMap


def assess_md_production_acceptance(
    postprocess_report_payload: JsonMap,
    minimum_incidents: int = PRODUCTION_MIN_INCIDENTS,
) -> MDProductionAcceptanceReport:
    blockers: list[str] = []
    evidence: list[str] = []
    event_count = _positive_int(postprocess_report_payload, "event_count")
    expected_count = _positive_int(postprocess_report_payload, "expected_incident_count")
    errors = _string_list(postprocess_report_payload, "errors")

    if postprocess_report_payload.get("ok") is not True:
        blockers.append("md_postprocess_not_ok")
    if postprocess_report_payload.get("postprocess_status") != "md_postprocess_complete":
        blockers.append("md_postprocess_not_complete")
    if postprocess_report_payload.get("verification_status") != "verified":
        blockers.append("md_verification_not_verified")
    if event_count < minimum_incidents:
        blockers.append(f"event_count_too_low:{event_count}<{minimum_incidents}")
    if expected_count < minimum_incidents:
        blockers.append(
            f"expected_incident_count_too_low:{expected_count}<{minimum_incidents}"
        )
    if _float_value(postprocess_report_payload, "total_deposited_energy_eV") <= 0.0:
        blockers.append("deposited_energy_not_positive")
    if errors:
        blockers.extend(f"md_postprocess_error:{error}" for error in errors)

    if not blockers:
        evidence.extend(("md_500_incidents_verified", "md_postprocess_accepted"))
    return MDProductionAcceptanceReport(
        accepted=not blockers,
        payload={
            "accepted": not blockers,
            "minimum_incidents": minimum_incidents,
            "event_count": event_count,
            "expected_incident_count": expected_count,
            "evidence": evidence,
            "blockers": blockers,
        },
    )


def _positive_int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def _float_value(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _string_list(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]
