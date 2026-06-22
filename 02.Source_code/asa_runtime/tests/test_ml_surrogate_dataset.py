from __future__ import annotations

from collections.abc import Mapping
import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
SUCCESS_LOG = FIXTURE_ROOT / "md_logs" / "success_lammps.log"
FAILED_LOG = FIXTURE_ROOT / "md_logs" / "failed_lammps.log"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _kernel_payload() -> Mapping[str, object]:
    return json.loads(KERNEL.read_text(encoding="utf-8"))


def test_verified_md_run_builds_surrogate_training_dataset() -> None:
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_training_dataset

    report = verify_md_run(SUCCESS_LOG, EVENTS, expected_events=2, required_ion="Ar", required_material="Si")
    spec = KernelFeatureSpec.from_mapping(_kernel_payload())

    dataset = build_training_dataset(report, spec)
    first = dataset.rows[0]
    second = dataset.rows[1]

    assert dataset.kernel_id == "Ar_on_Si__physical_fixture_v001"
    assert dataset.row_count == 2
    assert dataset.feature_columns == (
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
    assert first.feature_vector == pytest.approx(
        (100.0, 30.0, 120.0, 30.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.92, 0.08)
    )
    assert first.targets.sputter_yield_atoms_per_ion == pytest.approx(1.1)
    assert first.targets.deposited_energy_eV == pytest.approx(65.0)
    assert first.targets.reflection_probability == 0.0
    assert first.targets.implant_retained_fraction == pytest.approx(0.0)
    assert second.feature_vector == pytest.approx(
        (80.0, 45.0, 240.0, 45.0, 0.12, 0.14, 0.03, 1.5, 0.02, 1.0, 0.82, 0.18)
    )
    assert second.targets.reflection_probability == 1.0
    assert second.targets.reflection_energy_out_eV == pytest.approx(43.0)
    assert second.targets.reflection_polar_deg == pytest.approx(55.0)
    assert second.targets.reflection_azimuth_deg == pytest.approx(250.0)
    assert second.targets.implant_retained_fraction == pytest.approx(0.15)
    assert second.targets.implant_depth_mean_nm == pytest.approx(1.2)
    assert second.targets.removed_depth_nm == pytest.approx(0.005)


def test_surrogate_dataset_audit_records_quality_and_required_outputs() -> None:
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import (
        KernelFeatureSpec,
        audit_training_dataset,
        build_training_dataset,
    )

    report = verify_md_run(SUCCESS_LOG, EVENTS, expected_events=2, required_ion="Ar", required_material="Si")
    dataset = build_training_dataset(report, KernelFeatureSpec.from_mapping(_kernel_payload()))

    audit = audit_training_dataset(
        dataset,
        min_events=2,
        required_outputs=("energy_transfer", "sputtering", "reflection"),
    )

    assert audit.ok is True
    assert audit.payload["row_count"] == 2
    assert "dataset_rows_physically_sane" in audit.payload["evidence"]


def test_surrogate_dataset_audit_blocks_missing_required_output() -> None:
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import (
        KernelFeatureSpec,
        audit_training_dataset,
        build_training_dataset,
    )

    report = verify_md_run(SUCCESS_LOG, EVENTS, expected_events=2, required_ion="Ar", required_material="Si")
    dataset = build_training_dataset(report, KernelFeatureSpec.from_mapping(_kernel_payload()))

    audit = audit_training_dataset(dataset, min_events=2, required_outputs=("missing_output",))

    assert audit.ok is False
    assert "dataset_outputs_missing:missing_output" in audit.payload["blockers"]


def test_unverified_md_report_cannot_build_training_dataset() -> None:
    from sim_agent.md import verify_md_run
    from sim_agent.ml_surrogate import KernelFeatureSpec, SurrogateDatasetError, build_training_dataset

    report = verify_md_run(FAILED_LOG, EVENTS, expected_events=2)
    spec = KernelFeatureSpec.from_mapping(_kernel_payload())

    with pytest.raises(SurrogateDatasetError, match="verified_md_required"):
        build_training_dataset(report, spec)


def test_kernel_spec_requires_surface_state_features() -> None:
    from sim_agent.ml_surrogate import KernelFeatureSpec, SurrogateDatasetError

    payload = {
        "kernel_id": "bad_kernel",
        "ion_species": "Ar",
        "material_id": "Si",
        "inputs": ["energy_eV", "polar_deg", "azimuth_deg"],
        "outputs": ["energy_transfer"],
    }

    with pytest.raises(SurrogateDatasetError, match="missing_required_features"):
        KernelFeatureSpec.from_mapping(payload)


def test_build_surrogate_dataset_cli_reports_rows() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_surrogate_dataset.py"),
            "--log",
            str(SUCCESS_LOG),
            "--events",
            str(EVENTS),
            "--kernel",
            str(KERNEL),
            "--expected-events",
            "2",
            "--required-ion",
            "Ar",
            "--required-material",
            "Si",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "surrogate_dataset_ok=true" in result.stdout
    assert "row_count=2" in result.stdout
    assert "feature_columns=energy_eV,polar_deg,azimuth_deg,local_incidence_deg" in result.stdout
    assert "total_removed_depth_nm=0.035" in result.stdout


def test_build_surrogate_dataset_cli_blocks_failed_md() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "build_surrogate_dataset.py"),
            "--log",
            str(FAILED_LOG),
            "--events",
            str(EVENTS),
            "--kernel",
            str(KERNEL),
            "--expected-events",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "surrogate_dataset_ok=false" in result.stdout
    assert "verified_md_required" in result.stdout
