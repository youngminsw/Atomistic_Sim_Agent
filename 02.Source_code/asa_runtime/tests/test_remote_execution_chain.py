from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence
from sim_agent.compute import WorkerBundle


def test_remote_execution_chain_preserves_campaign_then_lammps_order(tmp_path: Path) -> None:
    from sim_agent.compute import build_remote_execution_chain

    campaign_worker, lammps_worker, postprocess_worker = _worker_bundles(tmp_path)

    chain = build_remote_execution_chain(
        (campaign_worker, lammps_worker, postprocess_worker),
        ssh_target="swym@10.24.12.85",
        ssh_port=55555,
    )

    assert tuple(stage.stage_id for stage in chain.stages) == (
        "01-plan-valid_ar_si_pr_hole-md-campaign",
        "02-plan-valid_ar_si_pr_hole-lammps-execution",
        "03-plan-valid_ar_si_pr_hole-md-postprocess",
    )
    assert "run_md_campaign_job.py" in chain.stages[0].plan.execution_command
    assert "run_lammps_execution_plan.py" in chain.stages[1].plan.execution_command
    assert "postprocess_lammps_execution.py" in chain.stages[2].plan.execution_command
    first_execution = chain.all_commands.index(chain.stages[0].plan.execution_command)
    second_execution = chain.all_commands.index(chain.stages[1].plan.execution_command)
    third_execution = chain.all_commands.index(chain.stages[2].plan.execution_command)
    assert first_execution < second_execution
    assert second_execution < third_execution


def test_render_remote_worker_chain_cli_writes_ordered_payload(tmp_path: Path) -> None:
    from sim_agent.compute import worker_bundle_payload

    campaign_worker, lammps_worker, postprocess_worker = _worker_bundles(tmp_path)
    campaign_path = tmp_path / "campaign_worker.json"
    lammps_path = tmp_path / "lammps_worker.json"
    postprocess_path = tmp_path / "postprocess_worker.json"
    out_path = tmp_path / "remote_chain.json"
    campaign_path.write_text(json.dumps(worker_bundle_payload(campaign_worker)), encoding="utf-8")
    lammps_path.write_text(json.dumps(worker_bundle_payload(lammps_worker)), encoding="utf-8")
    postprocess_path.write_text(
        json.dumps(worker_bundle_payload(postprocess_worker)),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "render_remote_worker_chain.py"),
            "--worker",
            str(campaign_path),
            "--worker",
            str(lammps_path),
            "--worker",
            str(postprocess_path),
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

    payload = as_mapping(json.loads(out_path.read_text(encoding="utf-8")), "chain")
    stages = as_sequence(payload["stages"], "stages")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "remote_worker_chain_ok=true" in result.stdout
    assert payload["stage_count"] == 3
    assert as_mapping(stages[0], "stage")["run_id"] == "plan-valid_ar_si_pr_hole-md-campaign"
    assert as_mapping(stages[1], "stage")["run_id"] == "plan-valid_ar_si_pr_hole-lammps-execution"
    assert as_mapping(stages[2], "stage")["run_id"] == "plan-valid_ar_si_pr_hole-md-postprocess"


def test_ui_http_writes_remote_execution_chain_when_ssh_target_is_present(
    tmp_path: Path,
) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    output_dir = tmp_path / "agent-plan"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = _post_json(
            f"http://{host}:{port}/api/agent/prepare-md-campaign-worker-bundle",
            {
                "request": _load_request("valid_ar_si_pr_hole.json"),
                "output_dir": str(output_dir),
                "host": "gpu-5090",
                "environment_name": "atomistic-sim-gpu",
                "ssh_target": "swym@10.24.12.85",
                "ssh_port": 55555,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    chain = as_mapping(body["remote_execution_chain"], "remote_execution_chain")
    worker = as_mapping(body["worker_bundle"], "worker_bundle")
    assert body["remote_execution_chain_path"] == str(output_dir / "remote_chain.json")
    assert body["remote_execution_script_path"] == str(output_dir / "remote_chain.sh")
    assert body["remote_execution_manifest_path"] == str(
        output_dir / "remote_chain_manifest.json"
    )
    assert chain["stage_count"] == 3
    assert (output_dir / "remote_chain.json").exists()
    script = (output_dir / "remote_chain.sh").read_text(encoding="utf-8")
    source_payload_path = output_dir / "source_payload.tar.gz"
    manifest = as_mapping(
        json.loads((output_dir / "remote_chain_manifest.json").read_text(encoding="utf-8")),
        "remote_chain_manifest",
    )
    worker_inputs = as_sequence(worker["input_paths"], "worker_inputs")
    assert "set -euo pipefail" in script
    assert 'cd "$SCRIPT_DIR"' in script
    assert "tar -xzf source_payload.tar.gz" in script
    assert "run_md_campaign_job.py" in script
    assert "run_lammps_execution_plan.py" in script
    assert "postprocess_lammps_execution.py" in script
    assert manifest["stage_count"] == 3
    assert manifest["executable_script"] == str(output_dir / "remote_chain.sh")
    assert "source_payload.tar.gz" in worker_inputs
    assert source_payload_path.exists()
    with tarfile.open(source_payload_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "02.Source_code/asa_runtime/scripts/run_md_campaign_job.py" in names
    assert "02.Source_code/asa_runtime/scripts/probe_worker_capability.py" in names
    assert "02.Source_code/asa_runtime/sim_agent/__init__.py" in names
    assert (
        "02.Source_code/asa_runtime/tests/fixtures/materials/si_amorphous_descriptor.json"
        in names
    )


def _worker_bundles(tmp_path: Path) -> tuple[WorkerBundle, WorkerBundle, WorkerBundle]:
    from sim_agent.compute import (
        build_lammps_execution_job_from_md_campaign_job,
        build_md_campaign_job_from_plan_dir,
        build_md_postprocess_job_from_lammps_execution_job,
        build_worker_bundle,
    )

    _write_plan_artifacts(tmp_path)
    campaign_job = build_md_campaign_job_from_plan_dir(tmp_path, "atomistic-sim-gpu")
    lammps_job = build_lammps_execution_job_from_md_campaign_job(campaign_job)
    postprocess_job = build_md_postprocess_job_from_lammps_execution_job(
        lammps_job,
        material_id="Si",
    )
    return (
        build_worker_bundle("gpu-5090", campaign_job, remote_user="swym"),
        build_worker_bundle("gpu-5090", lammps_job, remote_user="swym"),
        build_worker_bundle("gpu-5090", postprocess_job, remote_user="swym"),
    )


def _write_plan_artifacts(output_dir: Path) -> None:
    from sim_agent.agent_harness import (
        OfflineModelClient,
        SimulationAgentHarness,
        write_agent_plan_artifacts,
    )
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    endpoint = ModelProviderConfig.from_mapping(
        as_mapping(payload["llm_endpoint"], "llm_endpoint")
    )
    result = SimulationAgentHarness(
        endpoint=endpoint,
        client=OfflineModelClient(),
    ).plan(payload)
    write_agent_plan_artifacts(output_dir, payload, result)


def _post_json(url: str, payload: JsonMap) -> JsonMap:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-ASA-CSRF-Token": "test-token"},
        method="POST",
    )
    response = urlopen(request, timeout=5)
    return as_mapping(json.loads(response.read().decode("utf-8")), "response")


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)
