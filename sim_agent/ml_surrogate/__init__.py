from __future__ import annotations

from .dataset import SurrogateDatasetAuditReport, audit_training_dataset, build_training_dataset
from .empirical_mdn import (
    EMPIRICAL_MDN_ARTIFACT,
    EMPIRICAL_MDN_BACKEND,
    EmpiricalMDNModel,
    empirical_mdn_payload,
    write_empirical_mdn_model,
)
from .kernel import (
    CoverageRange,
    InteractionContext,
    InteractionKernel,
    InteractionKernelError,
    InteractionKernelManifest,
    InteractionKernelRegistry,
    KernelCoverage,
    KernelInferenceReport,
    build_fixture_interaction_kernel,
)
from .training_gate import (
    DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION,
    MDNTrainingMetrics,
    SurrogateTrainingCriteria,
    SurrogateTrainingGateReport,
    assess_surrogate_training_readiness,
    surrogate_training_gate_report_payload,
)
from .registry import (
    SurrogateModelRegistration,
    SurrogateModelRegistryError,
    register_surrogate_model,
)
from .uncertainty_map import UncertaintyMapSample, uncertainty_map_payload, write_uncertainty_map
from .types import (
    KernelFeatureSpec,
    SurrogateDatasetError,
    SurrogateTargets,
    SurrogateTrainingDataset,
    SurrogateTrainingRow,
)

__all__ = [
    "CoverageRange",
    "InteractionContext",
    "InteractionKernel",
    "InteractionKernelError",
    "InteractionKernelManifest",
    "InteractionKernelRegistry",
    "KernelCoverage",
    "KernelFeatureSpec",
    "KernelInferenceReport",
    "DEFAULT_MAX_HIGH_UNCERTAINTY_FRACTION",
    "EMPIRICAL_MDN_ARTIFACT",
    "EMPIRICAL_MDN_BACKEND",
    "EmpiricalMDNModel",
    "MDNTrainingMetrics",
    "SurrogateDatasetAuditReport",
    "SurrogateDatasetError",
    "SurrogateModelRegistration",
    "SurrogateModelRegistryError",
    "SurrogateTargets",
    "SurrogateTrainingCriteria",
    "SurrogateTrainingDataset",
    "SurrogateTrainingGateReport",
    "SurrogateTrainingRow",
    "UncertaintyMapSample",
    "audit_training_dataset",
    "build_fixture_interaction_kernel",
    "build_training_dataset",
    "empirical_mdn_payload",
    "register_surrogate_model",
    "assess_surrogate_training_readiness",
    "surrogate_training_gate_report_payload",
    "uncertainty_map_payload",
    "write_empirical_mdn_model",
    "write_uncertainty_map",
]
