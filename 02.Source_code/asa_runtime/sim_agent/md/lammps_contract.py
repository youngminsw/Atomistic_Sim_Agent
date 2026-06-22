from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.materials import ForceFieldRecord
from sim_agent.schemas._parse import JsonMap


class LAMMPSContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LAMMPSUnitSystem:
    unit_style: str
    distance_unit: str
    time_unit: str
    energy_unit: str
    timestep_unit: str


@dataclass(frozen=True, slots=True)
class LAMMPSOutputSpec:
    filename: str
    artifact_type: str
    parser_role: str


@dataclass(frozen=True, slots=True)
class CollisionTreatment:
    zbl_required: bool
    high_energy_collision_model: str
    force_field_protocol_id: str
    potential_name: str
    force_field_source_url: str


@dataclass(frozen=True, slots=True)
class LAMMPSOutputValidation:
    ok: bool
    missing_filenames: tuple[str, ...]
    found_filenames: tuple[str, ...]
    error_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LAMMPSOutputContract:
    run_id: str
    unit_system: LAMMPSUnitSystem
    output_specs: tuple[LAMMPSOutputSpec, ...]
    collision_treatment: CollisionTreatment

    @property
    def required_filenames(self) -> tuple[str, ...]:
        return tuple(spec.filename for spec in self.output_specs)

    def validate_output_dir(self, run_dir: Path) -> LAMMPSOutputValidation:
        found = tuple(filename for filename in self.required_filenames if (run_dir / filename).exists())
        missing = tuple(filename for filename in self.required_filenames if filename not in found)
        return LAMMPSOutputValidation(
            ok=not missing,
            missing_filenames=missing,
            found_filenames=found,
            error_lines=tuple(f"missing:{filename}" for filename in missing),
        )

    def manifest_payload(self) -> JsonMap:
        return {
            "run_id": self.run_id,
            "unit_style": self.unit_system.unit_style,
            "distance_unit": self.unit_system.distance_unit,
            "time_unit": self.unit_system.time_unit,
            "energy_unit": self.unit_system.energy_unit,
            "timestep_unit": self.unit_system.timestep_unit,
            "required_outputs": self.required_filenames,
            "zbl_required": self.collision_treatment.zbl_required,
            "high_energy_collision_model": self.collision_treatment.high_energy_collision_model,
            "force_field_protocol_id": self.collision_treatment.force_field_protocol_id,
            "force_field_source_url": self.collision_treatment.force_field_source_url,
        }


DEFAULT_UNIT_SYSTEM: Final = LAMMPSUnitSystem(
    unit_style="metal",
    distance_unit="angstrom",
    time_unit="ps",
    energy_unit="eV",
    timestep_unit="ps",
)

DEFAULT_OUTPUT_SPECS: Final = (
    LAMMPSOutputSpec("run_manifest.json", "manifest", "run metadata and units"),
    LAMMPSOutputSpec("surface_snapshot_before.data", "structure", "pre-impact active surface"),
    LAMMPSOutputSpec("surface_snapshot_after.data", "structure", "post-impact active surface"),
    LAMMPSOutputSpec("incident.dump", "trajectory", "incident ion states"),
    LAMMPSOutputSpec("reflected.dump", "trajectory", "reflected or scattered outgoing states"),
    LAMMPSOutputSpec("sputtered.dump", "trajectory", "species-resolved sputtered atoms"),
    LAMMPSOutputSpec("implanted.dump", "trajectory", "retained or implanted projectiles"),
    LAMMPSOutputSpec("traj.dump", "trajectory", "compact trajectory diagnostics"),
    LAMMPSOutputSpec("energy_depth_profile.csv", "profile", "depth-resolved deposited energy"),
    LAMMPSOutputSpec("damage_profile.csv", "profile", "damage and amorphization descriptors"),
    LAMMPSOutputSpec("roughness_rdf_descriptor.json", "descriptor", "roughness RDF and order descriptors"),
    LAMMPSOutputSpec("log.lammps", "log", "LAMMPS completion and thermo evidence"),
)


def build_lammps_output_contract(run_id: str, force_field: ForceFieldRecord) -> LAMMPSOutputContract:
    if not force_field.protocol_id or not force_field.source_url:
        raise LAMMPSContractError("force_field_provenance_required")
    return LAMMPSOutputContract(
        run_id=run_id,
        unit_system=DEFAULT_UNIT_SYSTEM,
        output_specs=DEFAULT_OUTPUT_SPECS,
        collision_treatment=CollisionTreatment(
            zbl_required=force_field.zbl_required,
            high_energy_collision_model="zbl_overlay" if force_field.zbl_required else "force_field_only",
            force_field_protocol_id=force_field.protocol_id,
            potential_name=force_field.potential_name,
            force_field_source_url=force_field.source_url,
        ),
    )
