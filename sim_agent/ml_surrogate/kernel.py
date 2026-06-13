from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.md import MDRunStatus, MDVerificationReport, inspect_md_events
from sim_agent.schemas.common import UncertaintyReport
from sim_agent.schemas.events import (
    DamageDelta,
    EnergyDepthPoint,
    EnergyTransfer,
    EventBundle,
    ImplantationOutcome,
    ReflectionOutcome,
    SputteringOutcome,
)

from .coverage import CoverageRange, KernelCoverage, coverage_from_dataset, uncertainty_for_context
from .dataset import build_training_dataset
from .types import KernelFeatureSpec, SurrogateDatasetError, SurrogateTrainingDataset, SurrogateTrainingRow


DEFAULT_FORCE_FIELD_PROTOCOL_ID: Final = "Si_Tersoff_ZBL_physical_v001"
DEFAULT_PHYSICS_SCOPE: Final = "physical_bombardment_no_chemistry"


@dataclass(frozen=True, slots=True)
class InteractionKernelError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class InteractionContext:
    ion_species: str
    material_id: str
    force_field_protocol_id: str
    physics_scope: str
    energy_eV: float
    polar_deg: float
    azimuth_deg: float
    local_incidence_deg: float
    phase: str
    amorphous_index: float
    roughness_rms_nm: float
    rdf_crystal_similarity: float
    rdf_amorphous_similarity: float
    damage_dose: float
    implanted_inert_fraction: float
    local_fluence: float
    removed_depth_nm: float


@dataclass(frozen=True, slots=True)
class InteractionKernelManifest:
    kernel_id: str
    ion_species: str
    material_id: str
    force_field_protocol_id: str
    physics_scope: str
    training_event_count: int
    coverage: KernelCoverage
    provenance_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KernelInferenceReport:
    bundle: EventBundle
    active_learning_suggested: bool


@dataclass(frozen=True, slots=True)
class InteractionKernel:
    manifest: InteractionKernelManifest
    dataset: SurrogateTrainingDataset

    def registry(self) -> InteractionKernelRegistry:
        return InteractionKernelRegistry(kernels=(self,))

    def sample(self, context: InteractionContext) -> EventBundle:
        return self.infer(context).bundle

    def infer(self, context: InteractionContext) -> KernelInferenceReport:
        row = _nearest_row(self.dataset, context)
        uncertainty = uncertainty_for_context(self.manifest.coverage, context)
        bundle = _bundle_from_row(row, context, uncertainty)
        return KernelInferenceReport(bundle=bundle, active_learning_suggested=uncertainty.ood)


@dataclass(frozen=True, slots=True)
class InteractionKernelRegistry:
    kernels: tuple[InteractionKernel, ...]

    def select(self, context: InteractionContext) -> InteractionKernel:
        for kernel in self.kernels:
            if _same_expert(kernel.manifest, context):
                return kernel
        raise InteractionKernelError("new_campaign_required")

    def infer(self, context: InteractionContext) -> KernelInferenceReport:
        return self.select(context).infer(context)


def build_fixture_interaction_kernel(
    events_path: Path,
    spec: KernelFeatureSpec,
    provenance_source: str,
) -> InteractionKernel:
    check = inspect_md_events(
        events_path, expected_events=None, required_ion=spec.ion_species, required_material=spec.material_id
    )
    if check.errors or check.dataset is None:
        raise SurrogateDatasetError("valid_fixture_events_required")
    report = MDVerificationReport(
        ok=True,
        status=MDRunStatus.VERIFIED,
        dataset=check.dataset,
        evidence=("fixture_events_verified",),
        errors=(),
    )
    dataset = build_training_dataset(report, spec)
    return InteractionKernel(
        manifest=InteractionKernelManifest(
            kernel_id=dataset.kernel_id,
            ion_species=spec.ion_species,
            material_id=spec.material_id,
            force_field_protocol_id=DEFAULT_FORCE_FIELD_PROTOCOL_ID,
            physics_scope=DEFAULT_PHYSICS_SCOPE,
            training_event_count=dataset.row_count,
            coverage=coverage_from_dataset(dataset),
            provenance_sources=(provenance_source,),
        ),
        dataset=dataset,
    )


def _same_expert(manifest: InteractionKernelManifest, context: InteractionContext) -> bool:
    return (
        manifest.ion_species == context.ion_species
        and manifest.material_id == context.material_id
        and manifest.force_field_protocol_id == context.force_field_protocol_id
        and manifest.physics_scope == context.physics_scope
    )


def _nearest_row(dataset: SurrogateTrainingDataset, context: InteractionContext) -> SurrogateTrainingRow:
    return min(
        dataset.rows,
        key=lambda row: abs(row.feature_vector[0] - context.energy_eV)
        + abs(row.feature_vector[1] - context.polar_deg)
        + 0.1 * abs(row.feature_vector[2] - context.azimuth_deg),
    )


def _bundle_from_row(
    row: SurrogateTrainingRow,
    context: InteractionContext,
    uncertainty: UncertaintyReport,
) -> EventBundle:
    target = row.targets
    return EventBundle(
        event_type_probabilities={
            "reflect": target.reflection_probability,
            "sputter": target.sputter_probability,
        },
        reflection=ReflectionOutcome(
            probability=target.reflection_probability,
            energy_out_eV=target.reflection_energy_out_eV,
            polar_deg=target.reflection_polar_deg,
            azimuth_deg=target.reflection_azimuth_deg,
        ),
        sputtering=SputteringOutcome(
            yield_atoms_per_ion=target.sputter_yield_atoms_per_ion,
            species_yields={context.material_id: target.sputter_yield_atoms_per_ion},
        ),
        energy_transfer=EnergyTransfer(
            deposited_energy_eV=target.deposited_energy_eV,
            depth_profile=(EnergyDepthPoint(depth_nm=0.5, energy_eV=target.deposited_energy_eV),),
            lateral_moment_nm=context.roughness_rms_nm,
        ),
        implantation=ImplantationOutcome(
            retained_fraction=target.implant_retained_fraction,
            depth_mean_nm=target.implant_depth_mean_nm,
        ),
        damage_delta=DamageDelta(
            amorphous_index=min(1.0, context.amorphous_index + target.removed_depth_nm),
            damage_dose=context.damage_dose + target.deposited_energy_eV,
            roughness_rms_nm=context.roughness_rms_nm + target.removed_depth_nm,
        ),
        removed_depth_nm=target.removed_depth_nm,
        uncertainty=uncertainty,
    )
