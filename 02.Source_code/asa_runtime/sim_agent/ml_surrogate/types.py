from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap, as_sequence, as_str, str_field


REQUIRED_FEATURES: Final = (
    "energy_eV",
    "polar_deg",
    "azimuth_deg",
    "local_incidence_deg",
    "amorphous_index",
    "roughness_rms_nm",
    "removed_depth_nm",
    "damage_dose",
    "implanted_inert_fraction",
    "local_fluence",
    "rdf_crystal_similarity",
    "rdf_amorphous_similarity",
)
KNOWN_FEATURES: Final = frozenset(REQUIRED_FEATURES)


class SurrogateDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class KernelFeatureSpec:
    kernel_id: str
    ion_species: str
    material_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: JsonMap) -> KernelFeatureSpec:
        inputs = tuple(as_str(item, "inputs[]") for item in as_sequence(value.get("inputs", ()), "inputs"))
        outputs = tuple(as_str(item, "outputs[]") for item in as_sequence(value.get("outputs", ()), "outputs"))
        _validate_inputs(inputs)
        return cls(
            kernel_id=str_field(value, "kernel_id"),
            ion_species=str_field(value, "ion_species"),
            material_id=str_field(value, "material_id"),
            inputs=inputs,
            outputs=outputs,
        )


@dataclass(frozen=True, slots=True)
class SurrogateTargets:
    reflection_probability: float
    sputter_probability: float
    sputter_yield_atoms_per_ion: float
    reflection_energy_out_eV: float
    reflection_polar_deg: float
    reflection_azimuth_deg: float
    implant_retained_fraction: float
    implant_depth_mean_nm: float
    deposited_energy_eV: float
    removed_depth_nm: float


@dataclass(frozen=True, slots=True)
class SurrogateTrainingRow:
    event_id: str
    ion: str
    material_id: str
    feature_vector: tuple[float, ...]
    targets: SurrogateTargets


@dataclass(frozen=True, slots=True)
class SurrogateTrainingDataset:
    kernel_id: str
    feature_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    rows: tuple[SurrogateTrainingRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def total_removed_depth_nm(self) -> float:
        return sum(row.targets.removed_depth_nm for row in self.rows)


def _validate_inputs(inputs: tuple[str, ...]) -> None:
    missing = tuple(feature for feature in REQUIRED_FEATURES if feature not in inputs)
    if missing:
        raise SurrogateDatasetError(f"missing_required_features={','.join(missing)}")
    unknown = tuple(feature for feature in inputs if feature not in KNOWN_FEATURES)
    if unknown:
        raise SurrogateDatasetError(f"unknown_features={','.join(unknown)}")
