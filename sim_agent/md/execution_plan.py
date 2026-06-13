from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import (
    JsonMap,
    as_bool,
    as_sequence,
    as_str,
    require,
)
from sim_agent.schemas.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionPlanError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class LAMMPSExecutionPlan:
    command_line: str
    manifest_payload: JsonMap


@dataclass(frozen=True, slots=True)
class _InputManifest:
    input_deck_id: str
    run_id: str
    surface_state_id: str
    incident_count: int
    required_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AssetsManifest:
    asset_manifest_id: str
    run_id: str
    surface_state_id: str
    assets_ready: bool
    structure_filename: str
    potential_filename: str


def build_lammps_execution_plan(
    input_manifest_payload: JsonMap,
    assets_manifest_payload: JsonMap,
    run_dir: Path,
    lammps_binary: str = "lmp",
) -> LAMMPSExecutionPlan:
    try:
        input_manifest = _input_manifest(input_manifest_payload)
        assets_manifest = _assets_manifest(assets_manifest_payload)
    except SchemaValidationError as exc:
        raise LAMMPSExecutionPlanError(str(exc)) from exc
    _ensure_same_run(input_manifest, assets_manifest)
    required_inputs = (
        "in.atomistic_campaign",
        assets_manifest.structure_filename,
        assets_manifest.potential_filename,
    )
    _ensure_files_exist(run_dir, required_inputs)
    command_line = _command_line(run_dir, lammps_binary)
    return LAMMPSExecutionPlan(
        command_line=command_line,
        manifest_payload=_manifest_payload(
            input_manifest,
            assets_manifest,
            run_dir,
            lammps_binary,
            command_line,
            required_inputs,
        ),
    )


def _input_manifest(payload: JsonMap) -> _InputManifest:
    return _InputManifest(
        input_deck_id=as_str(require(payload, "input_deck_id"), "input_deck_id"),
        run_id=as_str(require(payload, "run_id"), "run_id"),
        surface_state_id=as_str(require(payload, "surface_state_id"), "surface_state_id"),
        incident_count=_positive_int(payload, "incident_count"),
        required_outputs=tuple(
            as_str(item, "required_outputs")
            for item in as_sequence(require(payload, "required_outputs"), "required_outputs")
        ),
    )


def _assets_manifest(payload: JsonMap) -> _AssetsManifest:
    return _AssetsManifest(
        asset_manifest_id=as_str(require(payload, "asset_manifest_id"), "asset_manifest_id"),
        run_id=as_str(require(payload, "run_id"), "run_id"),
        surface_state_id=as_str(require(payload, "surface_state_id"), "surface_state_id"),
        assets_ready=as_bool(require(payload, "assets_ready"), "assets_ready"),
        structure_filename=as_str(require(payload, "structure_filename"), "structure_filename"),
        potential_filename=as_str(require(payload, "potential_filename"), "potential_filename"),
    )


def _ensure_same_run(input_manifest: _InputManifest, assets_manifest: _AssetsManifest) -> None:
    if input_manifest.run_id != assets_manifest.run_id:
        raise LAMMPSExecutionPlanError("execution_plan_run_id_mismatch")
    if input_manifest.surface_state_id != assets_manifest.surface_state_id:
        raise LAMMPSExecutionPlanError("execution_plan_surface_state_mismatch")
    if not assets_manifest.assets_ready:
        raise LAMMPSExecutionPlanError("lammps_assets_not_ready")


def _ensure_files_exist(run_dir: Path, filenames: tuple[str, ...]) -> None:
    missing = tuple(filename for filename in filenames if not (run_dir / filename).exists())
    if missing:
        raise LAMMPSExecutionPlanError(f"lammps_staged_input_missing={missing[0]}")


def _command_line(run_dir: Path, lammps_binary: str) -> str:
    return (
        f"cd {shlex.quote(str(run_dir))} && "
        f"{shlex.quote(lammps_binary)} -in in.atomistic_campaign"
    )


def _manifest_payload(
    input_manifest: _InputManifest,
    assets_manifest: _AssetsManifest,
    run_dir: Path,
    lammps_binary: str,
    command_line: str,
    required_inputs: tuple[str, ...],
) -> JsonMap:
    return {
        "execution_plan_id": f"{input_manifest.run_id}-execution-plan",
        "run_id": input_manifest.run_id,
        "input_deck_id": input_manifest.input_deck_id,
        "asset_manifest_id": assets_manifest.asset_manifest_id,
        "surface_state_id": input_manifest.surface_state_id,
        "execution_status": "ready_for_lammps",
        "preflight_ok": True,
        "execute_now": False,
        "working_directory": str(run_dir),
        "lammps_binary": lammps_binary,
        "input_deck": "in.atomistic_campaign",
        "command_line": command_line,
        "expected_incident_count": input_manifest.incident_count,
        "required_inputs": list(required_inputs),
        "expected_outputs": list(input_manifest.required_outputs),
    }


def _positive_int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise SchemaValidationError(f"{field} must be a positive integer")
