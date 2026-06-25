from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import as_mapping, as_sequence


def test_agent_cli_plans_amorphous_si_ar_pipeline(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "agent-cli-run"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Plan Ar etching on amorphous Si through the production MD pipeline",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_cli_ok=true" in result.stdout
    assert "pipeline_stage=agent_plan" in result.stdout
    assert "pipeline_stage=lammps_execution_worker_bundle" in result.stdout
    assert "amorphous_structure_prep_manifest_path=" in result.stdout
    assert "amorphous_structure_prep_worker_path=" in result.stdout
    assert "amorphous_structure_prep_remote_plan_path=" in result.stdout
    assert "graphdb_import_bundle_dir=" in result.stdout
    assert "graphdb_ingest_report_path=" in result.stdout
    request = as_mapping(
        json.loads((output_dir / "validated_request.json").read_text(encoding="utf-8")),
        "validated_request",
    )
    scene = as_mapping(request["scene"], "scene")
    surface = as_mapping(scene["surface_state"], "surface_state")
    plan = as_mapping(
        json.loads((output_dir / "md_campaign_plan.json").read_text(encoding="utf-8")),
        "md_campaign_plan",
    )
    lammps_worker = as_mapping(
        json.loads(
            (output_dir / "lammps_execution_worker_bundle.json").read_text(
                encoding="utf-8"
            )
        ),
        "lammps_execution_worker_bundle",
    )
    md_job = as_mapping(
        json.loads((output_dir / "md_campaign_job.json").read_text(encoding="utf-8")),
        "md_campaign_job",
    )
    prep_worker = as_mapping(
        json.loads(
            (output_dir / "amorphous_structure_prep_worker_bundle.json").read_text(
                encoding="utf-8"
            )
        ),
        "amorphous_structure_prep_worker",
    )
    requirements = as_mapping(
        lammps_worker["capability_requirements"],
        "capability_requirements",
    )
    md_box = as_mapping(surface["md_box"], "md_box")
    assert surface["material_id"] == "Si"
    assert surface["phase"] == "amorphous"
    assert surface["amorphous_index"] == 1.0
    assert md_box["atom_count"] == 5000
    assert md_box["run_length_ps"] == 2.0
    assert request["recipe"]["ion_species"] == "Ar"
    assert plan["phases"] == ["amorphous"]
    assert plan["energy_strata"]["minimum"] == 30.0
    assert plan["energy_strata"]["maximum"] == 150.0
    assert md_job["command"][-1] == "500"
    assert prep_worker["run_id"] == "plan-cli_ar_si_amorphous_hole-amorphous-structure-prep"
    assert as_sequence(requirements["required_lammps_packages"], "packages") == ["MANYBODY"]
    assert "--worker-capability" in " ".join(lammps_worker["command_line"].split())
    assert (
        output_dir / "amorphous_structure_prep" / "amorphous_structure_source.json"
    ).exists()
    assert (output_dir / "research_graph" / "ingest_report.json").exists()


def test_agent_cli_writes_remote_chain_script_when_ssh_target_is_given(
    tmp_path: Path,
) -> None:
    # Given
    output_dir = tmp_path / "agent-cli-remote-run"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Prepare remote execution for Ar etching on amorphous Si",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote_execution_script_path=" in result.stdout
    script_path = output_dir / "remote_chain.sh"
    manifest_path = output_dir / "remote_chain_manifest.json"
    assert script_path.exists()
    assert manifest_path.exists()
    script = script_path.read_text(encoding="utf-8")
    manifest = as_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "remote_chain_manifest",
    )
    assert "set -euo pipefail" in script
    assert "run_lammps_execution_plan.py" in script
    assert manifest["stage_count"] == 3


def test_resume_agent_run_from_request_injects_amorphous_structure_source(
    tmp_path: Path,
) -> None:
    initial_dir = tmp_path / "agent-cli-initial"
    resumed_dir = tmp_path / "agent-cli-resumed"
    initial = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Prepare amorphous Si structure before production MD",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--output-dir",
            str(initial_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    source_path = (
        initial_dir / "amorphous_structure_prep" / "amorphous_structure_source.json"
    )

    resumed = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "resume_agent_run_from_request.py"),
            "--request",
            str(initial_dir / "validated_request.json"),
            "--lammps-structure-source",
            str(source_path),
            "--output-dir",
            str(resumed_dir),
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    request = as_mapping(
        json.loads((resumed_dir / "validated_request.json").read_text(encoding="utf-8")),
        "validated_request",
    )
    surface = as_mapping(as_mapping(request["scene"], "scene")["surface_state"], "surface_state")
    source = as_mapping(surface["lammps_structure_source"], "lammps_structure_source")
    ledger = as_mapping(
        json.loads((resumed_dir / "agent_run_ledger.json").read_text(encoding="utf-8")),
        "agent_run_ledger",
    )

    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "resume_agent_run_ok=true" in resumed.stdout
    assert source["path"] == str(source_path)
    assert ledger["md"]["production_ready"] is True
    assert "amorphous_structure_source_present" in ledger["md"]["evidence"]
    assert ledger["compute_target"]["host"] == "gpu-5090"
    assert ledger["compute_target"]["ssh_target"] == "swym@10.24.12.85"
    assert (resumed_dir / "remote_chain_manifest.json").exists()


def test_agent_cli_can_run_remote_chain_and_record_failure(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "agent-cli-run-chain"
    env = _env_with_failing_ssh(tmp_path)

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
            "--offline",
            "--goal",
            "Run remote chain for Ar etching on amorphous Si",
            "--material",
            "Si",
            "--phase",
            "amorphous",
            "--ion",
            "Ar",
            "--feature-type",
            "hole",
            "--energy-range-eV",
            "30:150",
            "--polar-range-deg",
            "0:55",
            "--azimuth-range-deg",
            "0:360",
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--run-remote-chain",
            "--remote-run-timeout-s",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    result_path = output_dir / "remote_chain_result.json"
    payload = as_mapping(
        json.loads(result_path.read_text(encoding="utf-8")),
        "remote_chain_result",
    )
    assert result.returncode == 1
    assert "remote_chain_result_path=" in result.stdout
    assert payload["chain_status"] == "remote_chain_failed"
    assert "remote_chain_command_failed" in payload["blockers"]
    assert "Connection reset by peer" in payload["stderr_tail"]


def _env_with_failing_ssh(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Connection reset by peer' >&2\n"
        "exit 255\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    return env
