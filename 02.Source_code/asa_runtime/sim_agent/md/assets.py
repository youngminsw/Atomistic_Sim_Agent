from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse

from sim_agent.md.legacy_assets import (
    LEGACY_SI_CRYSTAL_STRUCTURE_SOURCE,
    LEGACY_SOURCE_PREFIX,
    resolve_repo_relative_path,
)
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str, require
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class LAMMPSAssetStagingError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class LAMMPSRunAssets:
    structure_path: Path
    potential_path: Path
    manifest_payload: JsonMap


@dataclass(frozen=True, slots=True)
class _AssetRequest:
    run_id: str
    material_id: str
    phase: str
    surface_state_id: str
    force_field_protocol_id: str
    force_field_source_url: str
    structure_source: JsonMap | None
    expected_atom_count: int | None


STRUCTURE_FILENAME: Final = "surface_snapshot_before.data"
POTENTIAL_FILENAME: Final = "Si.tersoff"
REPO_URL_PREFIX: Final = "repo://"
FILE_URL_PREFIX: Final = "file://"
SI_CRYSTAL_STRUCTURE_SOURCE: Final = LEGACY_SI_CRYSTAL_STRUCTURE_SOURCE


def stage_lammps_run_assets(
    contract_payload: JsonMap,
    surface_state_payload: JsonMap,
    output_dir: Path,
    repo_root: Path,
) -> LAMMPSRunAssets:
    try:
        request = _asset_request(contract_payload, surface_state_payload)
    except SchemaValidationError as exc:
        raise LAMMPSAssetStagingError(str(exc)) from exc
    structure_source = _structure_source(request, repo_root)
    _validate_structure_atom_count(structure_source, request.expected_atom_count)
    potential_source = _repo_source_path(request.force_field_source_url, repo_root)
    structure_path = output_dir / STRUCTURE_FILENAME
    potential_path = output_dir / POTENTIAL_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(structure_source, structure_path)
    shutil.copyfile(potential_source, potential_path)
    return LAMMPSRunAssets(
        structure_path=structure_path,
        potential_path=potential_path,
        manifest_payload=_manifest_payload(
            request,
            structure_source,
            potential_source,
            repo_root,
        ),
    )


def _asset_request(contract: JsonMap, surface: JsonMap) -> _AssetRequest:
    return _AssetRequest(
        run_id=as_str(require(contract, "run_id"), "run_id"),
        material_id=as_str(require(surface, "material_id"), "material_id"),
        phase=as_str(require(surface, "phase"), "phase"),
        surface_state_id=as_str(require(surface, "surface_state_id"), "surface_state_id"),
        force_field_protocol_id=as_str(
            require(contract, "force_field_protocol_id"),
            "force_field_protocol_id",
        ),
        force_field_source_url=as_str(
            require(contract, "force_field_source_url"),
            "force_field_source_url",
        ),
        structure_source=_optional_mapping(surface, "lammps_structure_source"),
        expected_atom_count=_md_box_atom_count(surface),
    )


def _structure_source(request: _AssetRequest, repo_root: Path) -> Path:
    if request.structure_source is not None:
        return _provided_structure_source(request, repo_root)
    if request.material_id == "Si" and request.phase == "crystal":
        return _repo_relative_path(SI_CRYSTAL_STRUCTURE_SOURCE, repo_root)
    if request.material_id == "Si" and request.phase == "amorphous":
        raise LAMMPSAssetStagingError("amorphous_lammps_structure_source_required")
    raise LAMMPSAssetStagingError("lammps_structure_fixture_unavailable")


def _repo_source_path(source_url: str, repo_root: Path) -> Path:
    if source_url.startswith(REPO_URL_PREFIX):
        return _repo_relative_path(source_url.removeprefix(REPO_URL_PREFIX), repo_root)
    raise LAMMPSAssetStagingError("repo_force_field_source_required")


def _provided_structure_source(request: _AssetRequest, repo_root: Path) -> Path:
    source = request.structure_source
    if source is None:
        raise LAMMPSAssetStagingError("lammps_structure_source_required")
    phase = as_str(require(source, "phase"), "lammps_structure_source.phase")
    if phase != request.phase:
        raise LAMMPSAssetStagingError("lammps_structure_source_phase_mismatch")
    preparation = as_str(
        require(source, "preparation"),
        "lammps_structure_source.preparation",
    )
    if request.phase == "amorphous" and not _relaxed_preparation(preparation):
        raise LAMMPSAssetStagingError("amorphous_structure_relaxation_required")
    path = _source_path(
        as_str(require(source, "path"), "lammps_structure_source.path"),
        repo_root,
    )
    if not path.exists():
        raise LAMMPSAssetStagingError("lammps_structure_source_not_found")
    return path


def _source_path(source_url: str, repo_root: Path) -> Path:
    if source_url.startswith(REPO_URL_PREFIX):
        return _repo_relative_path(source_url.removeprefix(REPO_URL_PREFIX), repo_root)
    if source_url.startswith(FILE_URL_PREFIX):
        parsed = urlparse(source_url)
        return Path(unquote(parsed.path))
    raise LAMMPSAssetStagingError("unsupported_lammps_structure_source")


def _repo_relative_path(relative: str, repo_root: Path) -> Path:
    return resolve_repo_relative_path(relative, repo_root)


def _relaxed_preparation(preparation: str) -> bool:
    normalized = preparation.lower()
    return "relax" in normalized or "import" in normalized


def _optional_mapping(payload: JsonMap, field: str) -> JsonMap | None:
    value = payload.get(field)
    if value is None:
        return None
    return dict(as_mapping(value, field))


def _md_box_atom_count(surface: JsonMap) -> int | None:
    value = surface.get("md_box")
    if value is None:
        return None
    md_box = as_mapping(value, "md_box")
    atom_count = md_box.get("atom_count")
    if isinstance(atom_count, int) and not isinstance(atom_count, bool) and atom_count > 0:
        return atom_count
    raise SchemaValidationError("md_box.atom_count must be a positive integer")


def _validate_structure_atom_count(path: Path, expected_atom_count: int | None) -> None:
    if expected_atom_count is None:
        return
    actual = _read_structure_atom_count(path)
    if actual is None:
        raise LAMMPSAssetStagingError("lammps_structure_atom_count_missing")
    if actual != expected_atom_count:
        raise LAMMPSAssetStagingError(
            f"lammps_structure_atom_count_mismatch:{actual}!={expected_atom_count}"
        )


def _read_structure_atom_count(path: Path) -> int | None:
    for raw_line in path.read_text(encoding="utf-8").splitlines()[:20]:
        parts = raw_line.strip().split()
        if len(parts) == 2 and parts[1] == "atoms":
            try:
                return int(parts[0])
            except ValueError:
                return None
    return None


def _manifest_payload(
    request: _AssetRequest,
    structure_source: Path,
    potential_source: Path,
    repo_root: Path,
) -> JsonMap:
    return {
        "asset_manifest_id": f"{request.run_id}-assets",
        "run_id": request.run_id,
        "surface_state_id": request.surface_state_id,
        "assets_ready": True,
        "material_id": request.material_id,
        "phase": request.phase,
        "structure_filename": STRUCTURE_FILENAME,
        "structure_source_kind": _structure_source_kind(request),
        "structure_source": _repo_uri(structure_source, repo_root),
        "expected_atom_count": request.expected_atom_count,
        "potential_filename": POTENTIAL_FILENAME,
        "potential_source_kind": "repo_force_field",
        "potential_source": _repo_uri(potential_source, repo_root),
        "force_field_protocol_id": request.force_field_protocol_id,
    }


def _structure_source_kind(request: _AssetRequest) -> str:
    if request.structure_source is None:
        return "repo_fixture"
    value = request.structure_source.get("kind")
    if isinstance(value, str) and value:
        return value
    return "provided_structure"


def _repo_uri(path: Path, repo_root: Path) -> str:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return str(path)
    return f"{REPO_URL_PREFIX}{relative.as_posix()}"
