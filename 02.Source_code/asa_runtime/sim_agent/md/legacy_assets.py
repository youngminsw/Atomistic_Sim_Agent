from __future__ import annotations

from pathlib import Path
from typing import Final


LEGACY_SOURCE_PREFIX: Final = "02.Source_code/mss_agent/"
LEGACY_SI_POTENTIAL_SOURCE: Final = (
    "02.Source_code/mss_agent/md_agent_window/Reference/force_field_library/"
    "potentials/Si.tersoff"
)
LEGACY_SI_CRYSTAL_STRUCTURE_SOURCE: Final = (
    "02.Source_code/mss_agent/md_agent_window/results/run_Ar_Si_3evts/"
    "Si_periodic.data"
)

_ASSET_ROOT: Final = Path(__file__).resolve().parent / "legacy_assets"
_RUNTIME_ASSETS: Final = {
    LEGACY_SI_POTENTIAL_SOURCE: _ASSET_ROOT / "Si.tersoff",
    LEGACY_SI_CRYSTAL_STRUCTURE_SOURCE: _ASSET_ROOT / "Si_periodic.data",
}


def resolve_repo_relative_path(relative: str, repo_root: Path) -> Path:
    candidate = repo_root / relative
    if candidate.exists() or not relative.startswith(LEGACY_SOURCE_PREFIX):
        return candidate

    stripped = relative.removeprefix(LEGACY_SOURCE_PREFIX)
    legacy_candidate = repo_root.parent / "mss_agent" / stripped
    if legacy_candidate.exists():
        return legacy_candidate

    runtime_asset = _RUNTIME_ASSETS.get(relative)
    if runtime_asset is not None:
        return runtime_asset

    return repo_root / stripped


def runtime_legacy_asset(relative: Path) -> Path | None:
    return _RUNTIME_ASSETS.get(f"02.Source_code/{relative.as_posix()}")
