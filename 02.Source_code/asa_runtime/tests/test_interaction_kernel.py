from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures"
EVENTS = FIXTURE_ROOT / "md_events" / "md_events_small.jsonl"
KERNEL = FIXTURE_ROOT / "kernels" / "offline_ar_si_kernel.json"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _registry():
    from sim_agent.ml_surrogate import KernelFeatureSpec, build_fixture_interaction_kernel

    spec = KernelFeatureSpec.from_mapping(json.loads(KERNEL.read_text(encoding="utf-8")))
    kernel = build_fixture_interaction_kernel(EVENTS, spec, provenance_source=str(EVENTS))
    return kernel.registry()


def _context(
    energy_eV: float = 100.0,
    polar_deg: float = 30.0,
    azimuth_deg: float = 120.0,
    material_id: str = "Si",
):
    from sim_agent.ml_surrogate import InteractionContext

    return InteractionContext(
        ion_species="Ar",
        material_id=material_id,
        force_field_protocol_id="Si_Tersoff_ZBL_physical_v001",
        physics_scope="physical_bombardment_no_chemistry",
        energy_eV=energy_eV,
        polar_deg=polar_deg,
        azimuth_deg=azimuth_deg,
        local_incidence_deg=polar_deg,
        phase="crystal",
        amorphous_index=0.0,
        roughness_rms_nm=0.1,
        rdf_crystal_similarity=0.92,
        rdf_amorphous_similarity=0.08,
        damage_dose=0.0,
        implanted_inert_fraction=0.0,
        local_fluence=0.0,
        removed_depth_nm=0.0,
    )


def test_registry_selects_kernel_by_material_ion_force_field_and_scope() -> None:
    registry = _registry()
    selected = registry.select(_context())

    assert selected.manifest.kernel_id == "Ar_on_Si__physical_fixture_v001"
    assert selected.manifest.material_id == "Si"
    assert selected.manifest.ion_species == "Ar"
    assert selected.manifest.force_field_protocol_id == "Si_Tersoff_ZBL_physical_v001"
    assert selected.manifest.physics_scope == "physical_bombardment_no_chemistry"


def test_kernel_returns_typed_event_bundle() -> None:
    from sim_agent.schemas.events import EventBundle

    inference = _registry().infer(_context())

    assert isinstance(inference.bundle, EventBundle)
    assert inference.bundle.energy_transfer.deposited_energy_eV == pytest.approx(65.0)
    assert inference.bundle.sputtering.yield_atoms_per_ion == pytest.approx(1.1)
    assert inference.bundle.uncertainty.ood is False
    assert inference.active_learning_suggested is False


def test_kernel_preserves_md_reflection_and_implantation_targets() -> None:
    inference = _registry().infer(_context(energy_eV=80.0, polar_deg=45.0))

    assert inference.bundle.reflection.energy_out_eV == pytest.approx(43.0)
    assert inference.bundle.reflection.polar_deg == pytest.approx(55.0)
    assert inference.bundle.reflection.azimuth_deg == pytest.approx(250.0)
    assert inference.bundle.implantation.retained_fraction == pytest.approx(0.15)
    assert inference.bundle.implantation.depth_mean_nm == pytest.approx(1.2)


def test_ood_context_reports_uncertainty_and_active_learning() -> None:
    inference = _registry().infer(_context(energy_eV=2000.0, polar_deg=80.0))

    assert inference.bundle.uncertainty.ood is True
    assert inference.bundle.uncertainty.score > 0.5
    assert inference.active_learning_suggested is True


def test_ood_azimuth_context_reports_uncertainty_and_active_learning() -> None:
    inference = _registry().infer(_context(energy_eV=100.0, polar_deg=30.0, azimuth_deg=20.0))

    assert inference.bundle.uncertainty.ood is True
    assert inference.active_learning_suggested is True


def test_new_material_requires_new_campaign() -> None:
    from sim_agent.ml_surrogate import InteractionKernelError

    with pytest.raises(InteractionKernelError, match="new_campaign_required"):
        _registry().select(_context(material_id="PR"))


def test_kernel_manifest_records_coverage_and_provenance() -> None:
    manifest = _registry().select(_context()).manifest

    assert manifest.training_event_count == 2
    assert manifest.coverage.energy_eV.minimum == pytest.approx(80.0)
    assert manifest.coverage.energy_eV.maximum == pytest.approx(100.0)
    assert manifest.coverage.polar_deg.maximum == pytest.approx(45.0)
    assert manifest.provenance_sources == (str(EVENTS),)


def test_smoke_physics_kernel_cli_samples_event_bundle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_physics_kernel.py"),
            "--fixture",
            str(EVENTS),
            "--context",
            "ar_si_100ev_30deg",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "event_bundle_valid=true" in result.stdout
    assert "reflection_probability=" in result.stdout
    assert "sputtering_yield=" in result.stdout
    assert "energy_transfer_eV=65.0" in result.stdout
    assert "uncertainty_score=" in result.stdout


def test_smoke_physics_kernel_cli_reports_ood_active_learning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_physics_kernel.py"),
            "--fixture",
            str(EVENTS),
            "--context",
            "ar_si_2000ev_80deg",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ood=true" in result.stdout
    assert "active_learning_suggested=true" in result.stdout
