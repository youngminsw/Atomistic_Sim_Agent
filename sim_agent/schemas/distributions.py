from __future__ import annotations

from dataclasses import dataclass

from ._parse import JsonMap, as_mapping, as_sequence, float_field, str_field


@dataclass(frozen=True, slots=True)
class IonEnergyBin:
    min: float
    max: float
    probability: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> IonEnergyBin:
        return cls(
            min=float_field(value, "min"),
            max=float_field(value, "max"),
            probability=float_field(value, "probability"),
        )


@dataclass(frozen=True, slots=True)
class IonEnergyDistribution:
    kind: str
    unit: str
    bins: tuple[IonEnergyBin, ...]

    @classmethod
    def from_mapping(cls, value: JsonMap) -> IonEnergyDistribution:
        bins = tuple(IonEnergyBin.from_mapping(as_mapping(item, "bins[]")) for item in as_sequence(value.get("bins", []), "bins"))
        return cls(kind=str_field(value, "kind"), unit=str_field(value, "unit"), bins=bins)


@dataclass(frozen=True, slots=True)
class IonAngularDistribution:
    kind: str
    polar_min_deg: float
    polar_max_deg: float
    azimuth_min_deg: float
    azimuth_max_deg: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> IonAngularDistribution:
        return cls(
            kind=str_field(value, "kind"),
            polar_min_deg=float_field(value, "polar_min_deg"),
            polar_max_deg=float_field(value, "polar_max_deg"),
            azimuth_min_deg=float_field(value, "azimuth_min_deg"),
            azimuth_max_deg=float_field(value, "azimuth_max_deg"),
        )


@dataclass(frozen=True, slots=True)
class FluxSegment:
    start_s: float
    end_s: float
    flux: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> FluxSegment:
        return cls(start_s=float_field(value, "start_s"), end_s=float_field(value, "end_s"), flux=float_field(value, "flux"))


@dataclass(frozen=True, slots=True)
class FluxSchedule:
    unit: str
    segments: tuple[FluxSegment, ...]

    @classmethod
    def from_mapping(cls, value: JsonMap) -> FluxSchedule:
        segments = tuple(
            FluxSegment.from_mapping(as_mapping(item, "segments[]"))
            for item in as_sequence(value.get("segments", []), "segments")
        )
        return cls(unit=str_field(value, "unit"), segments=segments)


@dataclass(frozen=True, slots=True)
class SpeciesFraction:
    species: str
    fraction: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> SpeciesFraction:
        return cls(species=str_field(value, "species"), fraction=float_field(value, "fraction"))


@dataclass(frozen=True, slots=True)
class SpeciesMix:
    entries: tuple[SpeciesFraction, ...]

    @classmethod
    def from_sequence(cls, value: object) -> SpeciesMix:
        return cls(
            entries=tuple(
                SpeciesFraction.from_mapping(as_mapping(item, "species_mix[]"))
                for item in as_sequence(value, "species_mix")
            )
        )
