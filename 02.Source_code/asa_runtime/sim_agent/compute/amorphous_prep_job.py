from __future__ import annotations

from .source_payload import SOURCE_PAYLOAD_ARCHIVE
from .types import ComputePolicyError, JobBundleSpec


def build_amorphous_structure_prep_job(
    run_id: str,
    environment_name: str,
    material_id: str,
    atom_count: int,
    lammps_binary: str = "lmp",
) -> JobBundleSpec:
    if material_id != "Si":
        raise ComputePolicyError("amorphous_structure_prep_only_si_supported")
    if atom_count <= 0:
        raise ComputePolicyError("amorphous_structure_prep_atom_count_must_be_positive")
    job_id = f"{run_id}-amorphous-structure-prep"
    artifact_dir = f"artifacts/{job_id}"
    return JobBundleSpec(
        job_id=job_id,
        environment_name=environment_name,
        command=(
            "python3",
            "02.Source_code/asa_runtime/scripts/prepare_amorphous_structure_job.py",
            "--material",
            material_id,
            "--atom-count",
            str(atom_count),
            "--out-dir",
            artifact_dir,
            "--lammps-binary",
            lammps_binary,
            "--execute",
        ),
        input_paths=(SOURCE_PAYLOAD_ARCHIVE,),
        output_paths=(
            f"{artifact_dir}/amorphous_structure_prep_manifest.json",
            f"{artifact_dir}/amorphous_structure_source.json",
            f"{artifact_dir}/in.amorphous_prep",
            f"{artifact_dir}/Si.tersoff",
            f"{artifact_dir}/a_si_melt_quench_relaxed.data",
            f"{artifact_dir}/amorphous_structure_prep_result.json",
        ),
        requires_cuda=False,
    )
