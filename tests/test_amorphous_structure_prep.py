from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_prepare_amorphous_structure_job_cli_writes_lammps_prep_bundle(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "prepare_amorphous_structure_job.py"),
            "--material",
            "Si",
            "--atom-count",
            "5000",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(
        (tmp_path / "amorphous_structure_prep_manifest.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        (tmp_path / "amorphous_structure_source.json").read_text(encoding="utf-8")
    )
    deck = (tmp_path / "in.amorphous_prep").read_text(encoding="utf-8")

    assert "amorphous_structure_prep_ok=true" in result.stdout
    assert manifest["material_id"] == "Si"
    assert manifest["target_atom_count"] == 5000
    assert manifest["execution_required"] is True
    assert manifest["requires_user_or_scheduler_approval"] is True
    assert manifest["expected_output_structure"] == "a_si_melt_quench_relaxed.data"
    assert "lmp -in in.amorphous_prep" in manifest["command_line"]
    assert source["kind"] == "agent_prepared"
    assert source["phase"] == "amorphous"
    assert source["preparation"] == "melt_quench_relaxed"
    assert source["path"].endswith("/a_si_melt_quench_relaxed.data")
    assert "create_atoms 1 random 5000" in deck
    assert "fix melt all nvt temp 3500.000 3500.000" in deck
    assert "fix quench all nvt temp 3500.000 300.000" in deck
    assert "write_data a_si_melt_quench_relaxed.data" in deck
    assert (tmp_path / "Si.tersoff").exists()
