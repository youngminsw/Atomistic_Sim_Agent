from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "jobs"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_compute_policy_remote_gpu_priority_then_local_fallback() -> None:
    from sim_agent.compute import allowed_compute_hosts, select_compute_target

    hosts = allowed_compute_hosts()
    assert hosts[:3] == ("gpu-5090", "blackwell-rtxpro", "gpu-ada")
    assert "4090-gpu-ws" in hosts
    assert "ws-gpu" in hosts
    assert "local-rtx4060" in hosts
    assert "ws-24core" in hosts

    target = select_compute_target(("ws-24core", "gpu-ada", "gpu-5090"))

    assert target.host_alias == "gpu-5090"
    assert target.remote is True
    assert target.uses_local_fallback is False


def test_compute_policy_falls_back_to_local_when_remote_gpu_is_unavailable() -> None:
    from sim_agent.compute import select_compute_target

    target = select_compute_target(("ws-24core", "orca"), allow_local_fallback=True)

    assert target.host_alias == "local-rtx4060"
    assert target.remote is False
    assert target.uses_local_fallback is True


def test_worker_inventory_resolves_5090_default_connection() -> None:
    from sim_agent.compute import resolve_worker_host

    host = resolve_worker_host(
        host_alias="gpu-5090",
        environment_name="atomistic-sim-gpu",
        remote_user=None,
        ssh_target=None,
        ssh_port=None,
    )

    assert host.host_alias == "gpu-5090"
    assert host.environment_name == "atomistic-sim-gpu"
    assert host.remote_user == "swym"
    assert host.ssh_target == "swym@10.24.12.85"
    assert host.ssh_port == 55555
    assert host.inventory_source == "runtime_config"


def test_worker_bundle_uses_user_scoped_paths_and_conda_environment() -> None:
    from sim_agent.compute import build_worker_bundle, load_job_bundle

    job = load_job_bundle(FIXTURE_ROOT / "tiny_gpu_job.json")
    bundle = build_worker_bundle("gpu-5090", job, remote_user="swym")

    assert bundle.host_alias == "gpu-5090"
    assert bundle.environment_name == "atomistic-sim-gpu"
    assert str(bundle.remote_run_dir).startswith("/home/swym/atomistic_sim_agent/runs/tiny_gpu_job")
    assert "conda run -n atomistic-sim-gpu" in bundle.command_line
    assert "sudo" not in bundle.command_line
    assert bundle.preflight_commands == (
        "command -v conda",
        "conda env list | grep atomistic-sim-gpu",
        "nvidia-smi",
        (
            "cd /home/swym/atomistic_sim_agent/runs/tiny_gpu_job && "
            "python3 02.Source_code/mss_agent/scripts/probe_worker_capability.py "
            "--host gpu-5090 --environment-name atomistic-sim-gpu "
            "--artifact-root /home/swym/atomistic_sim_agent/runs/tiny_gpu_job "
            "--out worker_capability.json --requires-cuda"
        ),
    )
    assert bundle.capability_manifest_path == "worker_capability.json"
    remote_root = "/home/swym/atomistic_sim_agent/runs/tiny_gpu_job"
    expected_upload = (
        "upload:tests/fixtures/kernels/offline_ar_si_kernel.json->"
        f"{remote_root}/tests/fixtures/kernels/offline_ar_si_kernel.json"
    )
    expected_download = (
        f"download:{remote_root}/artifacts/tiny_gpu_job/manifest.json->"
        "artifacts/tiny_gpu_job/manifest.json"
    )
    assert bundle.input_paths == ("tests/fixtures/kernels/offline_ar_si_kernel.json",)
    assert bundle.output_paths == ("artifacts/tiny_gpu_job/manifest.json",)
    assert bundle.transfer_plan == (expected_upload, expected_download)


def test_cpu_md_job_can_run_on_allowed_gpu_host_without_cuda_preflight() -> None:
    from sim_agent.compute import JobBundleSpec, build_worker_bundle

    job = JobBundleSpec(
        job_id="tiny_md_cpu_job",
        environment_name="atomistic-sim-gpu",
        command=("lmp", "-in", "in.fixture"),
        input_paths=("md/in.fixture",),
        output_paths=("md/log.lammps",),
        requires_cuda=False,
    )

    bundle = build_worker_bundle("ws-gpu", job, remote_user="swym")

    assert bundle.host_alias == "ws-gpu"
    assert bundle.preflight_commands == (
        "command -v conda",
        "conda env list | grep atomistic-sim-gpu",
        (
            "cd /home/swym/atomistic_sim_agent/runs/tiny_md_cpu_job && "
            "python3 02.Source_code/mss_agent/scripts/probe_worker_capability.py "
            "--host ws-gpu --environment-name atomistic-sim-gpu "
            "--artifact-root /home/swym/atomistic_sim_agent/runs/tiny_md_cpu_job "
            "--out worker_capability.json --requires-lammps "
            "--required-lammps-packages MANYBODY"
        ),
    )
    assert "lmp -in in.fixture" in bundle.command_line


def test_compute_policy_cli_reports_allowed_hosts_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "check_compute_policy.py"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "host=gpu-5090" in result.stdout
    assert "host=blackwell-rtxpro" in result.stdout
    assert "host=gpu-ada" in result.stdout
    assert "host=ws-gpu" in result.stdout
    assert "host=local-rtx4060" in result.stdout
    assert "host=ws-24core" in result.stdout


def test_cpu_only_alias_is_rejected_by_worker_bundle_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_worker_bundle.py"),
            "--host",
            "ws-24core",
            "--job",
            str(FIXTURE_ROOT / "tiny_gpu_job.json"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "host_role_not_allowed=ws-24core:requires_gpu" in result.stdout


def test_worker_bundle_cli_prints_preflight_and_transfer_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_worker_bundle.py"),
            "--host",
            "gpu-5090",
            "--job",
            str(FIXTURE_ROOT / "tiny_gpu_job.json"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight=nvidia-smi" in result.stdout
    assert "transfer=upload:tests/fixtures/kernels/offline_ar_si_kernel.json->" in result.stdout
    download_prefix = (
        "transfer=download:/home/swym/atomistic_sim_agent/runs/tiny_gpu_job/"
        "artifacts/tiny_gpu_job/manifest.json->"
    )
    assert download_prefix in result.stdout


def test_worker_bundle_cli_writes_json_plan_for_remote_executor(tmp_path: Path) -> None:
    out_path = tmp_path / "worker_bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "run_worker_bundle.py"),
            "--host",
            "gpu-5090",
            "--job",
            str(FIXTURE_ROOT / "tiny_gpu_job.json"),
            "--dry-run",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["host_alias"] == "gpu-5090"
    assert payload["environment_name"] == "atomistic-sim-gpu"
    assert payload["remote_run_dir"] == "/home/swym/atomistic_sim_agent/runs/tiny_gpu_job"
    assert "conda run -n atomistic-sim-gpu" in payload["command_line"]
    assert "nvidia-smi" in payload["preflight_commands"]
    assert payload["preflight_commands"][-1].endswith("--requires-cuda")
    assert payload["capability_manifest_path"] == "worker_capability.json"
    assert payload["input_paths"] == ["tests/fixtures/kernels/offline_ar_si_kernel.json"]
    assert payload["output_paths"] == ["artifacts/tiny_gpu_job/manifest.json"]
    assert payload["transfer_plan"][0].startswith(
        "upload:tests/fixtures/kernels/offline_ar_si_kernel.json->"
    )
