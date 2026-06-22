from __future__ import annotations

import tarfile
from dataclasses import dataclass
from os import walk
from pathlib import Path
from threading import Lock

from sim_agent.schemas._parse import JsonMap

from .types import ComputePolicyError


SOURCE_PAYLOAD_ARCHIVE = "source_payload.tar.gz"
SOURCE_PAYLOAD_ROOT = "02.Source_code"
SOURCE_PAYLOAD_ENTRIES = (
    Path("asa_runtime/scripts/prepare_amorphous_structure_job.py"),
    Path("asa_runtime/scripts/probe_worker_capability.py"),
    Path("asa_runtime/scripts/run_md_campaign_job.py"),
    Path("asa_runtime/scripts/run_lammps_execution_plan.py"),
    Path("asa_runtime/scripts/postprocess_lammps_execution.py"),
    Path("mss_agent/md_agent_window/Reference/force_field_library/potentials/Si.tersoff"),
    Path("mss_agent/md_agent_window/results/run_Ar_Si_3evts/Si_periodic.data"),
    Path("asa_runtime/sim_agent"),
    Path("asa_runtime/tests/fixtures/materials"),
)


@dataclass(frozen=True, slots=True)
class SourcePayloadBundle:
    archive_path: Path
    manifest_payload: JsonMap


@dataclass(frozen=True, slots=True)
class SourcePayloadSnapshot:
    archive_bytes: bytes
    entries: tuple[str, ...]


_SOURCE_PAYLOAD_CACHE: dict[Path, SourcePayloadSnapshot] = {}
_SOURCE_PAYLOAD_CACHE_LOCK = Lock()


def stage_compute_source_payload(source_root: Path, output_dir: Path) -> SourcePayloadBundle:
    archive_path = output_dir / SOURCE_PAYLOAD_ARCHIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = _stage_archive(_source_code_root(source_root), archive_path)
    manifest = {
        "archive_path": str(archive_path),
        "archive_name": SOURCE_PAYLOAD_ARCHIVE,
        "payload_root": SOURCE_PAYLOAD_ROOT,
        "entry_count": len(copied_paths),
        "entries": copied_paths,
    }
    return SourcePayloadBundle(archive_path=archive_path, manifest_payload=manifest)


def _stage_archive(source_root: Path, archive_path: Path) -> list[str]:
    snapshot = _cached_snapshot(source_root)
    if snapshot is not None:
        archive_path.write_bytes(snapshot.archive_bytes)
        return list(snapshot.entries)
    copied_paths: list[str] = []
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for entry in SOURCE_PAYLOAD_ENTRIES:
            copied_paths.extend(_add_entry(archive, source_root, entry))
    snapshot = SourcePayloadSnapshot(
        archive_bytes=archive_path.read_bytes(),
        entries=tuple(copied_paths),
    )
    with _SOURCE_PAYLOAD_CACHE_LOCK:
        _SOURCE_PAYLOAD_CACHE.setdefault(source_root, snapshot)
    return copied_paths


def _source_code_root(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if (resolved / "asa_runtime").is_dir() and (resolved / "mss_agent").is_dir():
        return resolved
    if resolved.name == "asa_runtime" and (resolved.parent / "mss_agent").is_dir():
        return resolved.parent
    return resolved


def _cached_snapshot(source_root: Path) -> SourcePayloadSnapshot | None:
    with _SOURCE_PAYLOAD_CACHE_LOCK:
        return _SOURCE_PAYLOAD_CACHE.get(source_root)


def _add_entry(tar: tarfile.TarFile, source_root: Path, relative_root: Path) -> list[str]:
    root = source_root / relative_root
    if not root.exists():
        raise ComputePolicyError(f"source_payload_missing={relative_root.as_posix()}")
    if root.is_file():
        archive_name = _archive_name(source_root, root)
        tar.add(root, arcname=archive_name)
        return [archive_name]
    added: list[str] = []
    for path in _iter_payload_files(root):
        archive_name = _archive_name(source_root, path)
        tar.add(path, arcname=archive_name)
        added.append(archive_name)
    return added


def _iter_payload_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in walk(root):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname != "__pycache__")
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if _should_include(path):
                files.append(path)
    return tuple(files)


def _archive_name(source_root: Path, path: Path) -> str:
    return f"{SOURCE_PAYLOAD_ROOT}/{path.relative_to(source_root).as_posix()}"


def _should_include(path: Path) -> bool:
    if not path.is_file():
        return False
    if "__pycache__" in path.parts:
        return False
    if path.name.endswith(":Zone.Identifier"):
        return False
    return path.suffix not in {".pyc", ".pyo"}
