from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_write_potential_acquisition_ledger_preserves_sandbox_run(
    tmp_path: Path,
) -> None:
    from sim_agent.materials import (
        PotentialAcquisitionRequest,
        acquire_potential_candidate,
        write_potential_acquisition_ledger,
    )

    # Given
    potential_path, metadata_path = _write_valid_tersoff_fixture(tmp_path)
    fake_lammps = _write_fake_lammps(tmp_path)
    report = acquire_potential_candidate(
        PotentialAcquisitionRequest(
            source_url=potential_path.as_uri(),
            metadata_url=metadata_path.as_uri(),
            material_id="Si",
            ion_species="Ar",
            required_elements=("Si", "Ar"),
            sandbox_command=(sys.executable, str(fake_lammps)),
            sandbox_work_dir=tmp_path / "ledger" / "sandbox_smoke",
        ),
        repo_root=SOURCE_ROOT,
    )

    # When
    ledger = write_potential_acquisition_ledger(
        output_dir=tmp_path / "ledger",
        run_id="potential-run-001",
        report=report,
    )

    # Then
    payload = as_mapping(json.loads(ledger.ledger_path.read_text(encoding="utf-8")), "ledger")
    artifacts = as_mapping(payload["artifacts"], "artifacts")
    sandbox = as_mapping(payload["sandbox_smoke"], "sandbox_smoke")
    assert ledger.artifact_count == 5
    assert payload["run_id"] == "potential-run-001"
    assert payload["gate_status"] == "potential_candidate_accepted"
    assert artifacts["sandbox_input"] == "sandbox_smoke/in.potential_smoke"
    assert artifacts["sandbox_potential"] == "sandbox_smoke/potential_under_test.ff"
    assert sandbox["smoke_status"] == "sandbox_smoke_passed"
    assert (tmp_path / "ledger" / "potential_acquisition_report.json").exists()
    assert (tmp_path / "ledger" / "potential_candidate.json").exists()
    assert (tmp_path / "ledger" / "potential_validation_report.json").exists()
    assert (tmp_path / "ledger" / "sandbox_smoke_report.json").exists()
    assert (tmp_path / "ledger" / "sandbox_smoke" / "in.potential_smoke").exists()


def test_acquire_potential_candidate_cli_writes_ledger_dir(tmp_path: Path) -> None:
    # Given
    potential_path, metadata_path = _write_valid_tersoff_fixture(tmp_path)
    fake_lammps = _write_fake_lammps(tmp_path)
    report_path = tmp_path / "report.json"
    ledger_dir = tmp_path / "ledger"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "acquire_potential_candidate.py"),
            "--source-url",
            potential_path.as_uri(),
            "--metadata-url",
            metadata_path.as_uri(),
            "--material",
            "Si",
            "--ion",
            "Ar",
            "--required-elements",
            "Si,Ar",
            "--lammps-command",
            f"{sys.executable} {fake_lammps}",
            "--ledger-dir",
            str(ledger_dir),
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    ledger = as_mapping(
        json.loads((ledger_dir / "ledger.json").read_text(encoding="utf-8")),
        "ledger",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ledger_path=" in result.stdout
    assert ledger["run_id"] == "potential-acquisition"
    assert ledger["gate_status"] == "potential_candidate_accepted"
    assert (ledger_dir / "sandbox_smoke" / "potential_under_test.ff").exists()


def _write_valid_tersoff_fixture(root: Path) -> tuple[Path, Path]:
    potential_root = root / "potentials"
    potential_root.mkdir()
    potential_path = potential_root / "Si.tersoff"
    metadata_path = potential_root / "Si.tersoff.metadata.json"
    potential_path.write_text("# elements: Si Ar\nSi Si Si 3.0\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(_valid_tersoff_metadata()), encoding="utf-8")
    return (potential_path, metadata_path)


def _valid_tersoff_metadata() -> JsonMap:
    return {
        "potential_id": "si-tersoff-zbl-ledger-v001",
        "material_id": "Si",
        "ion_species": "Ar",
        "pair_style": "hybrid/overlay tersoff zbl",
        "potential_name": "Ledger Si.tersoff + ZBL overlay",
        "provenance_url": "https://github.com/lammps/lammps/blob/develop/potentials/README",
        "license": "repo-test",
        "lammps_unit_style": "metal",
        "atom_type_mapping": ["Si", "Ar"],
        "fitted_system": "Si collision cascade fixture with ZBL overlay for Ar projectile",
        "transferability_scope": "Ar physical sputtering of Si with ZBL high-energy term",
    }


def _write_fake_lammps(root: Path) -> Path:
    script = root / "fake_lammps_ok.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "deck = Path(sys.argv[sys.argv.index('-in') + 1])\n"
        "(deck.parent / 'log.lammps').write_text('Loop time of 0.01 on 1 procs\\n')\n"
        "print('Loop time of 0.01 on 1 procs')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    return script
