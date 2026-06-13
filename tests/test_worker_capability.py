from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence


def test_validate_worker_capability_accepts_lammps_gpu_manifest() -> None:
    from sim_agent.compute import (
        validate_worker_capability,
        worker_capability_requirements_payload,
    )

    # Given
    requirements = worker_capability_requirements_payload(
        host_alias="gpu-5090",
        environment_name="atomistic-sim-gpu",
        remote_run_dir="/home/swym/atomistic_sim_agent/runs/lammps-job",
        requires_cuda=True,
        command=("python3", "02.Source_code/mss_agent/scripts/run_lammps_execution_plan.py"),
    )

    # When
    report = validate_worker_capability(_capability_manifest(), requirements)

    # Then
    assert report.ok is True
    assert report.payload["gate_status"] == "worker_capability_ready"
    assert "conda_environment_present" in report.payload["evidence"]
    assert "lammps_required_packages_present" in report.payload["evidence"]
    assert "gpu_capability_present" in report.payload["evidence"]


def test_validate_worker_capability_rejects_missing_lammps_package() -> None:
    from sim_agent.compute import (
        validate_worker_capability,
        worker_capability_requirements_payload,
    )

    # Given
    manifest = _capability_manifest(lammps_packages=("KSPACE",))
    requirements = worker_capability_requirements_payload(
        host_alias="gpu-5090",
        environment_name="atomistic-sim-gpu",
        remote_run_dir="/home/swym/atomistic_sim_agent/runs/lammps-job",
        requires_cuda=False,
        command=("python3", "02.Source_code/mss_agent/scripts/run_lammps_execution_plan.py"),
    )

    # When
    report = validate_worker_capability(manifest, requirements)

    # Then
    assert report.ok is False
    assert report.payload["gate_status"] == "worker_capability_rejected"
    assert "lammps_required_package_missing:MANYBODY" in report.payload["blockers"]


def test_lammps_worker_bundle_payload_includes_capability_contract(tmp_path: Path) -> None:
    from sim_agent.compute import (
        build_lammps_execution_job_from_md_campaign_job,
        build_md_campaign_job_from_plan_dir,
        build_worker_bundle,
        worker_bundle_payload,
    )

    # Given
    _write_plan_artifacts(tmp_path)
    campaign_job = build_md_campaign_job_from_plan_dir(tmp_path, "atomistic-sim-gpu")
    lammps_job = build_lammps_execution_job_from_md_campaign_job(campaign_job)
    bundle = build_worker_bundle("gpu-5090", lammps_job, remote_user="swym")

    # When
    payload = worker_bundle_payload(bundle)

    # Then
    requirements = as_mapping(payload["capability_requirements"], "capability_requirements")
    required_packages = as_sequence(
        requirements["required_lammps_packages"],
        "required_lammps_packages",
    )
    assert payload["capability_manifest_path"] == "worker_capability.json"
    assert requirements["requires_lammps"] is True
    assert required_packages == ["MANYBODY"]
    assert "probe_worker_capability.py" in "\n".join(payload["preflight_commands"])


def test_probe_worker_capability_cli_writes_validated_report(tmp_path: Path) -> None:
    # Given
    fake_conda = _write_fake_conda(tmp_path)
    fake_lammps = _write_fake_lammps_help(tmp_path)
    report_path = tmp_path / "worker_capability.json"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "probe_worker_capability.py"),
            "--host",
            "local",
            "--environment-name",
            "atomistic-sim-gpu",
            "--artifact-root",
            str(tmp_path),
            "--lammps-binary",
            str(fake_lammps),
            "--requires-lammps",
            "--required-lammps-packages",
            "MANYBODY",
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        env=_probe_env(tmp_path, fake_conda),
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(report_path.read_text(encoding="utf-8")), "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker_capability_ok=true" in result.stdout
    assert payload["gate_status"] == "worker_capability_ready"


def _capability_manifest(lammps_packages: tuple[str, ...] = ("MANYBODY", "REAXFF")) -> JsonMap:
    return {
        "host_alias": "gpu-5090",
        "hostname": "gpu5090ws",
        "environment_name": "atomistic-sim-gpu",
        "conda_available": True,
        "conda_environment_present": True,
        "python_executable": "/home/swym/miniconda3/envs/atomistic-sim-gpu/bin/python",
        "python_version": "3.12.4",
        "artifact_root": "/home/swym/atomistic_sim_agent/runs/lammps-job",
        "artifact_root_writable": True,
        "gpu_available": True,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "cuda_visible": True,
        "lammps_available": True,
        "lammps_executable": "/home/swym/miniconda3/envs/atomistic-sim-gpu/bin/lmp",
        "lammps_version": "LAMMPS 22 Jul 2025",
        "lammps_packages": list(lammps_packages),
    }


def _write_plan_artifacts(output_dir: Path) -> None:
    from sim_agent.agent_harness import (
        OfflineModelClient,
        SimulationAgentHarness,
        write_agent_plan_artifacts,
    )
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = as_mapping(
        json.loads((REQUEST_ROOT / "valid_ar_si_pr_hole.json").read_text(encoding="utf-8")),
        "valid_ar_si_pr_hole",
    )
    endpoint = ModelProviderConfig.from_mapping(
        as_mapping(payload["llm_endpoint"], "llm_endpoint")
    )
    result = SimulationAgentHarness(
        endpoint=endpoint,
        client=OfflineModelClient(),
    ).plan(payload)
    write_agent_plan_artifacts(output_dir, payload, result)


def _write_fake_lammps_help(root: Path) -> Path:
    script = root / "fake_lammps_help.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('LAMMPS 22 Jul 2025')\n"
        "print('Installed packages: MANYBODY KSPACE')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _write_fake_conda(root: Path) -> Path:
    script = root / "conda"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('# conda environments:')\n"
        "print('atomistic-sim-gpu  /tmp/atomistic-sim-gpu')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _probe_env(root: Path, fake_conda: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{fake_conda.parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["CONDA_DEFAULT_ENV"] = "atomistic-sim-gpu"
    return env
