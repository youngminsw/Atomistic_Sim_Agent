from __future__ import annotations

from dataclasses import dataclass

from ._parse import JsonMap, as_mapping, as_sequence, float_field, float_map, optional_str, str_field
from .common import UncertaintyReport


@dataclass(frozen=True, slots=True)
class ReflectionOutcome:
    probability: float
    energy_out_eV: float
    polar_deg: float
    azimuth_deg: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> ReflectionOutcome:
        return cls(
            probability=float_field(value, "probability"),
            energy_out_eV=float_field(value, "energy_out_eV"),
            polar_deg=float_field(value, "polar_deg"),
            azimuth_deg=float_field(value, "azimuth_deg"),
        )


@dataclass(frozen=True, slots=True)
class SputteringOutcome:
    yield_atoms_per_ion: float
    species_yields: dict[str, float]

    @classmethod
    def from_mapping(cls, value: JsonMap) -> SputteringOutcome:
        return cls(
            yield_atoms_per_ion=float_field(value, "yield_atoms_per_ion"),
            species_yields=float_map(value.get("species_yields", {}), "species_yields"),
        )


@dataclass(frozen=True, slots=True)
class EnergyDepthPoint:
    depth_nm: float
    energy_eV: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> EnergyDepthPoint:
        return cls(depth_nm=float_field(value, "depth_nm"), energy_eV=float_field(value, "energy_eV"))


@dataclass(frozen=True, slots=True)
class EnergyTransfer:
    deposited_energy_eV: float
    depth_profile: tuple[EnergyDepthPoint, ...]
    lateral_moment_nm: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> EnergyTransfer:
        return cls(
            deposited_energy_eV=float_field(value, "deposited_energy_eV"),
            depth_profile=tuple(
                EnergyDepthPoint.from_mapping(as_mapping(item, "depth_profile[]"))
                for item in as_sequence(value.get("depth_profile", []), "depth_profile")
            ),
            lateral_moment_nm=float_field(value, "lateral_moment_nm"),
        )


@dataclass(frozen=True, slots=True)
class ImplantationOutcome:
    retained_fraction: float
    depth_mean_nm: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> ImplantationOutcome:
        return cls(retained_fraction=float_field(value, "retained_fraction"), depth_mean_nm=float_field(value, "depth_mean_nm"))


@dataclass(frozen=True, slots=True)
class DamageDelta:
    amorphous_index: float
    damage_dose: float
    roughness_rms_nm: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> DamageDelta:
        return cls(
            amorphous_index=float_field(value, "amorphous_index"),
            damage_dose=float_field(value, "damage_dose"),
            roughness_rms_nm=float_field(value, "roughness_rms_nm"),
        )


@dataclass(frozen=True, slots=True)
class EventBundle:
    event_type_probabilities: dict[str, float]
    reflection: ReflectionOutcome
    sputtering: SputteringOutcome
    energy_transfer: EnergyTransfer
    implantation: ImplantationOutcome
    damage_delta: DamageDelta
    removed_depth_nm: float
    uncertainty: UncertaintyReport

    @classmethod
    def from_mapping(cls, value: object) -> EventBundle:
        mapping = as_mapping(value, "event_bundle")
        return cls(
            event_type_probabilities=float_map(mapping.get("event_type_probabilities", {}), "event_type_probabilities"),
            reflection=ReflectionOutcome.from_mapping(as_mapping(mapping.get("reflection"), "reflection")),
            sputtering=SputteringOutcome.from_mapping(as_mapping(mapping.get("sputtering"), "sputtering")),
            energy_transfer=EnergyTransfer.from_mapping(as_mapping(mapping.get("energy_transfer"), "energy_transfer")),
            implantation=ImplantationOutcome.from_mapping(as_mapping(mapping.get("implantation"), "implantation")),
            damage_delta=DamageDelta.from_mapping(as_mapping(mapping.get("damage_delta"), "damage_delta")),
            removed_depth_nm=float_field(mapping, "removed_depth_nm"),
            uncertainty=UncertaintyReport.from_mapping(as_mapping(mapping.get("uncertainty"), "uncertainty")),
        )


@dataclass(frozen=True, slots=True)
class EnergyField:
    field_id: str
    units: str


@dataclass(frozen=True, slots=True)
class DamageField:
    field_id: str
    units: str


@dataclass(frozen=True, slots=True)
class ProfileState:
    step: int
    artifact_path: str


@dataclass(frozen=True, slots=True)
class MDEvent:
    event_id: str
    bundle: EventBundle
    source_run_id: str | None = None

    @classmethod
    def from_mapping(cls, value: JsonMap) -> MDEvent:
        return cls(
            event_id=str_field(value, "event_id"),
            bundle=EventBundle.from_mapping(value.get("bundle")),
            source_run_id=optional_str(value, "source_run_id"),
        )
