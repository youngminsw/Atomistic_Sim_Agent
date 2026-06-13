from __future__ import annotations

from dataclasses import dataclass


class MaterialBuilderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MaterialDescriptor:
    structure_id: str
    material_id: str
    phase: str
    density_g_cm3: float
    rdf_order_features: dict[str, float]
    orientation: str
    preparation: str


@dataclass(frozen=True, slots=True)
class ForceFieldRecord:
    material_id: str
    ion_species: str
    protocol_id: str
    potential_name: str
    source_url: str
    zbl_required: bool


@dataclass(frozen=True, slots=True)
class PRMaterial:
    material_id: str
    role: str
    phase: str
    selectivity: float
    relative_erosion_rate: float


@dataclass(frozen=True, slots=True)
class MaterialBuildReport:
    material_id: str
    crystal: MaterialDescriptor | None
    amorphous: MaterialDescriptor | None
    pr_material: PRMaterial
    force_field: ForceFieldRecord
    dry_run: bool
