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

from sim_agent.schemas._parse import as_mapping


def test_remote_execution_plan_renders_ssh_and_rsync_commands() -> None:
    from sim_agent.compute import build_remote_execution_plan, build_worker_bundle, load_job_bundle

    job = load_job_bundle(FIXTURE_ROOT / "tiny_gpu_job.json")
    bundle = build_worker_bundle("gpu-5090", job, remote_user="swym")
    plan = build_remote_execution_plan(bundle, ssh_target="swym@10.24.12.85", ssh_port=55555)

    assert plan.ssh_target == "swym@10.24.12.85"
    assert plan.ssh_port == 55555
    assert plan.local_setup_commands == ("mkdir -p artifacts/tiny_gpu_job",)
    assert plan.remote_setup_commands[0].startswith("ssh -p 55555 swym@10.24.12.85 'mkdir -p ")
    assert plan.upload_commands[0].startswith(
        "rsync -az -e 'ssh -p 55555' tests/fixtures/kernels/offline_ar_si_kernel.json "
    )
    assert "ssh -p 55555 swym@10.24.12.85 nvidia-smi" in plan.preflight_commands
    assert "probe_worker_capability.py" in plan.preflight_commands[-1]
    assert "conda run -n atomistic-sim-gpu" in plan.execution_command
    assert plan.download_commands[0].startswith(
        "rsync -az -e 'ssh -p 55555' "
        "swym@10.24.12.85:/home/swym/atomistic_sim_agent/runs/tiny_gpu_job/"
        "worker_capability.json "
    )


def test_render_remote_worker_plan_cli_writes_command_payload(tmp_path: Path) -> None:
    from sim_agent.compute import build_worker_bundle, load_job_bundle, worker_bundle_payload

    job = load_job_bundle(FIXTURE_ROOT / "tiny_gpu_job.json")
    bundle = build_worker_bundle("gpu-5090", job, remote_user="swym")
    worker_path = tmp_path / "worker_bundle.json"
    out_path = tmp_path / "remote_plan.json"
    worker_path.write_text(json.dumps(worker_bundle_payload(bundle)), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "render_remote_worker_plan.py"),
            "--worker",
            str(worker_path),
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--out",
            str(out_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "remote_plan")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote_worker_plan_ok=true" in result.stdout
    assert payload["ssh_target"] == "swym@10.24.12.85"
    assert payload["ssh_port"] == 55555
    assert payload["execution_command"].startswith("ssh -p 55555 swym@10.24.12.85 ")
    assert "upload_commands" in payload
    assert "download_commands" in payload
