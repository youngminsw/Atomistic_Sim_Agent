from __future__ import annotations

import json
import math
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap


AVOGADRO_PER_MOL: Final = 6.022_140_76e23
SI_ATOMIC_MASS_G_PER_MOL: Final = 28.0855
SI_AMORPHOUS_DENSITY_G_CM3: Final = 2.28
SI_POTENTIAL_SOURCE: Final = (
    "02.Source_code/mss_agent/md_agent_window/Reference/force_field_library/potentials/Si.tersoff"
)
LEGACY_SOURCE_PREFIX: Final = "02.Source_code/mss_agent/"
POTENTIAL_FILENAME: Final = "Si.tersoff"
INPUT_FILENAME: Final = "in.amorphous_prep"
OUTPUT_STRUCTURE_FILENAME: Final = "a_si_melt_quench_relaxed.data"
MANIFEST_FILENAME: Final = "amorphous_structure_prep_manifest.json"
STRUCTURE_SOURCE_FILENAME: Final = "amorphous_structure_source.json"


@dataclass(frozen=True, slots=True)
class AmorphousStructurePrepError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class AmorphousStructurePrepConfig:
    material_id: str
    atom_count: int
    random_seed: int = 91_337
    melt_temp_k: float = 3500.0
    quench_temp_k: float = 300.0
    melt_steps: int = 20_000
    quench_steps: int = 50_000
    relax_steps: int = 10_000
    timestep_ps: float = 0.001
    density_g_cm3: float = SI_AMORPHOUS_DENSITY_G_CM3
    lammps_binary: str = "lmp"


@dataclass(frozen=True, slots=True)
class AmorphousStructurePrepBundle:
    manifest_path: Path
    structure_source_path: Path
    input_path: Path
    potential_path: Path
    manifest_payload: JsonMap
    structure_source_payload: JsonMap


def stage_amorphous_structure_prep_bundle(
    config: AmorphousStructurePrepConfig,
    output_dir: Path,
    repo_root: Path,
) -> AmorphousStructurePrepBundle:
    _ensure_supported(config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    potential_path = output_dir / POTENTIAL_FILENAME
    shutil.copyfile(_repo_relative_path(SI_POTENTIAL_SOURCE, repo_root), potential_path)
    input_path = output_dir / INPUT_FILENAME
    structure_output_path = output_dir / OUTPUT_STRUCTURE_FILENAME
    input_path.write_text(_input_script(config), encoding="utf-8")
    structure_source = _structure_source_payload(config, structure_output_path)
    manifest = _manifest_payload(config, output_dir, structure_source)
    manifest_path = output_dir / MANIFEST_FILENAME
    structure_source_path = output_dir / STRUCTURE_SOURCE_FILENAME
    _write_json(manifest_path, manifest)
    _write_json(structure_source_path, structure_source)
    return AmorphousStructurePrepBundle(
        manifest_path=manifest_path,
        structure_source_path=structure_source_path,
        input_path=input_path,
        potential_path=potential_path,
        manifest_payload=manifest,
        structure_source_payload=structure_source,
    )


def _ensure_supported(config: AmorphousStructurePrepConfig) -> None:
    if config.material_id != "Si":
        raise AmorphousStructurePrepError("amorphous_prep_only_si_supported")
    if config.atom_count <= 0:
        raise AmorphousStructurePrepError("atom_count_must_be_positive")
    if config.density_g_cm3 <= 0:
        raise AmorphousStructurePrepError("density_must_be_positive")


def _repo_relative_path(relative: str, repo_root: Path) -> Path:
    candidate = repo_root / relative
    if candidate.exists() or not relative.startswith(LEGACY_SOURCE_PREFIX):
        return candidate
    return repo_root / relative.removeprefix(LEGACY_SOURCE_PREFIX)


def _input_script(config: AmorphousStructurePrepConfig) -> str:
    box_length = _box_length_angstrom(config.atom_count, config.density_g_cm3)
    return "\n".join(
        (
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"region simbox block 0 {box_length:.6f} 0 {box_length:.6f} 0 {box_length:.6f} units box",
            "create_box 1 simbox",
            f"create_atoms 1 random {config.atom_count} {config.random_seed} simbox overlap 2.0 maxtry 100000",
            "mass 1 28.0855",
            "pair_style tersoff",
            f"pair_coeff * * {POTENTIAL_FILENAME} Si",
            f"timestep {config.timestep_ps:.6f}",
            "thermo 1000",
            f"velocity all create {config.melt_temp_k:.3f} {config.random_seed} mom yes rot yes dist gaussian",
            f"fix melt all nvt temp {config.melt_temp_k:.3f} {config.melt_temp_k:.3f} 0.100",
            f"run {config.melt_steps}",
            "unfix melt",
            f"fix quench all nvt temp {config.melt_temp_k:.3f} {config.quench_temp_k:.3f} 0.100",
            f"run {config.quench_steps}",
            "unfix quench",
            f"fix relax all nvt temp {config.quench_temp_k:.3f} {config.quench_temp_k:.3f} 0.100",
            f"run {config.relax_steps}",
            "unfix relax",
            "min_style cg",
            "minimize 1.0e-6 1.0e-8 1000 10000",
            f"write_data {OUTPUT_STRUCTURE_FILENAME}",
            "",
        )
    )


def _box_length_angstrom(atom_count: int, density_g_cm3: float) -> float:
    volume_cm3 = atom_count * SI_ATOMIC_MASS_G_PER_MOL / AVOGADRO_PER_MOL / density_g_cm3
    return math.pow(volume_cm3 * 1.0e24, 1.0 / 3.0)


def _manifest_payload(
    config: AmorphousStructurePrepConfig,
    output_dir: Path,
    structure_source: JsonMap,
) -> JsonMap:
    command_line = (
        f"cd {shlex.quote(str(output_dir))} && "
        f"{shlex.quote(config.lammps_binary)} -in {INPUT_FILENAME}"
    )
    return {
        "prep_manifest_id": f"a_si_melt_quench_relaxed_{config.atom_count}",
        "material_id": config.material_id,
        "phase": "amorphous",
        "preparation": "melt_quench_relaxed",
        "target_atom_count": config.atom_count,
        "density_g_cm3": config.density_g_cm3,
        "input_deck": INPUT_FILENAME,
        "potential_filename": POTENTIAL_FILENAME,
        "expected_output_structure": OUTPUT_STRUCTURE_FILENAME,
        "execution_required": True,
        "requires_user_or_scheduler_approval": True,
        "command_line": command_line,
        "structure_source": structure_source,
    }


def _structure_source_payload(config: AmorphousStructurePrepConfig, path: Path) -> JsonMap:
    return {
        "kind": "agent_prepared",
        "path": path.as_uri(),
        "phase": "amorphous",
        "preparation": "melt_quench_relaxed",
        "material_id": config.material_id,
        "atom_count": config.atom_count,
    }


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
