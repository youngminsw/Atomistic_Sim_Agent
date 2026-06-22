from __future__ import annotations

from pathlib import Path

from .events import inspect_md_events
from .logs import inspect_lammps_log
from .types import MDRunStatus, MDVerificationReport


def verify_md_run(
    log_path: Path,
    events_path: Path,
    expected_events: int | None = None,
    required_ion: str | None = None,
    required_material: str | None = None,
) -> MDVerificationReport:
    log_check = inspect_lammps_log(log_path)
    if log_check.errors:
        return MDVerificationReport(
            ok=False,
            status=MDRunStatus.FAILED,
            dataset=None,
            evidence=log_check.evidence,
            errors=log_check.errors,
        )
    event_check = inspect_md_events(events_path, expected_events, required_ion, required_material)
    if event_check.errors:
        return MDVerificationReport(
            ok=False,
            status=MDRunStatus.REJECTED,
            dataset=None,
            evidence=log_check.evidence,
            errors=event_check.errors,
        )
    return MDVerificationReport(
        ok=True,
        status=MDRunStatus.VERIFIED,
        dataset=event_check.dataset,
        evidence=log_check.evidence + event_check.evidence,
        errors=(),
    )
