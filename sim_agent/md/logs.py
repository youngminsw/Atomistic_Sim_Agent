from __future__ import annotations

from pathlib import Path

from .types import LammpsLogCheck


def inspect_lammps_log(path: Path) -> LammpsLogCheck:
    text = path.read_text(encoding="utf-8")
    evidence: list[str] = []
    errors: list[str] = []
    if "Total wall time:" in text and "Loop time" in text:
        evidence.append("lammps_completed")
    else:
        errors.append("lammps_incomplete")
    if "Lost atoms" in text:
        errors.append("lammps_lost_atoms")
    errors.extend(_generic_lammps_errors(text))
    return LammpsLogCheck(path=path, evidence=tuple(evidence), errors=tuple(dict.fromkeys(errors)))


def _generic_lammps_errors(text: str) -> tuple[str, ...]:
    errors: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("ERROR:") and "Lost atoms" not in line:
            errors.append("lammps_error")
    return tuple(errors)
