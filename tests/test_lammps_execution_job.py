from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_build_lammps_execution_job_from_md_campaign_job(tmp_path: Path) -> None:
    from sim_agent.compute import (
        build_lammps_execution_job_from_md_campaign_job,
        build_md_campaign_job_from_plan_dir,
        build_md_postprocess_job_from_lammps_execution_job,
        build_worker_bundle,
    )

    _write_plan_artifacts(tmp_path)
    campaign_job = build_md_campaign_job_from_plan_dir(tmp_path, "atomistic-sim-gpu")

    execution_job = build_lammps_execution_job_from_md_campaign_job(campaign_job)
    postprocess_job = build_md_postprocess_job_from_lammps_execution_job(
        execution_job,
        material_id="Si",
    )
    worker = build_worker_bundle("gpu-5090", execution_job, remote_user="swym")

    artifact_dir = "artifacts/plan-valid_ar_si_pr_hole-md-campaign"
    assert execution_job.job_id == "plan-valid_ar_si_pr_hole-lammps-execution"
    assert execution_job.command == (
        "python3",
        "02.Source_code/asa_runtime/scripts/run_lammps_execution_plan.py",
        "--plan",
        f"{artifact_dir}/lammps_execution_plan.json",
        "--out",
        f"{artifact_dir}/lammps_execution_result.json",
        "--worker-capability",
        "worker_capability.json",
        "--execute",
    )
    assert execution_job.input_paths == (
        "source_payload.tar.gz",
        f"{artifact_dir}/lammps_execution_plan.json",
        f"{artifact_dir}/in.atomistic_campaign",
        f"{artifact_dir}/surface_snapshot_before.data",
        f"{artifact_dir}/Si.tersoff",
    )
    assert execution_job.output_paths == _lammps_execution_outputs(artifact_dir)
    assert execution_job.requires_cuda is False
    assert "--incident-count" in campaign_job.command
    assert campaign_job.command[-1] == "500"
    assert worker.output_paths == execution_job.output_paths
    assert postprocess_job.job_id == "plan-valid_ar_si_pr_hole-md-postprocess"
    assert postprocess_job.command == (
        "python3",
        "02.Source_code/asa_runtime/scripts/postprocess_lammps_execution.py",
        "--execution-result",
        f"{artifact_dir}/lammps_execution_result.json",
        "--material",
        "Si",
        "--descriptor-root",
        "02.Source_code/asa_runtime/tests/fixtures/materials",
        "--events-out",
        f"{artifact_dir}/md_events.jsonl",
        "--report-out",
        f"{artifact_dir}/md_postprocess_report.json",
    )
    assert postprocess_job.input_paths == (
        "source_payload.tar.gz",
    ) + execution_job.output_paths
    assert postprocess_job.output_paths == (
        f"{artifact_dir}/md_events.jsonl",
        f"{artifact_dir}/md_postprocess_report.json",
    )


def test_prepare_md_campaign_worker_bundle_cli_writes_lammps_execution_job(
    tmp_path: Path,
) -> None:
    _write_plan_artifacts(tmp_path)
    lammps_job_path = tmp_path / "lammps_execution_job.json"
    lammps_worker_path = tmp_path / "lammps_execution_worker_bundle.json"
    postprocess_job_path = tmp_path / "md_postprocess_job.json"
    postprocess_worker_path = tmp_path / "md_postprocess_worker_bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "prepare_md_campaign_worker_bundle.py"),
            "--plan-dir",
            str(tmp_path),
            "--host",
            "gpu-5090",
            "--environment-name",
            "atomistic-sim-gpu",
            "--job-out",
            str(tmp_path / "md_campaign_job.json"),
            "--worker-out",
            str(tmp_path / "worker_bundle.json"),
            "--lammps-job-out",
            str(lammps_job_path),
            "--lammps-worker-out",
            str(lammps_worker_path),
            "--postprocess-job-out",
            str(postprocess_job_path),
            "--postprocess-worker-out",
            str(postprocess_worker_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "lammps_job_path=" in result.stdout
    assert "source_payload_path=" in result.stdout
    lammps_job = as_mapping(json.loads(lammps_job_path.read_text(encoding="utf-8")), "job")
    lammps_worker = as_mapping(
        json.loads(lammps_worker_path.read_text(encoding="utf-8")),
        "worker",
    )
    postprocess_job = as_mapping(
        json.loads(postprocess_job_path.read_text(encoding="utf-8")),
        "postprocess_job",
    )
    assert lammps_job["job_id"] == "plan-valid_ar_si_pr_hole-lammps-execution"
    assert lammps_worker["run_id"] == "plan-valid_ar_si_pr_hole-lammps-execution"
    assert postprocess_job["job_id"] == "plan-valid_ar_si_pr_hole-md-postprocess"


def test_ui_http_includes_lammps_execution_worker_bundle(tmp_path: Path) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host = server.server_name
    port = server.server_port
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
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    job = as_mapping(body["lammps_execution_job"], "lammps_execution_job")
    worker = as_mapping(body["lammps_execution_worker_bundle"], "worker")
    assert body["lammps_execution_job_path"] == str(output_dir / "lammps_execution_job.json")
    assert body["md_postprocess_job_path"] == str(output_dir / "md_postprocess_job.json")
    assert job["job_id"] == "plan-valid_ar_si_pr_hole-lammps-execution"
    assert worker["output_paths"] == job["outputs"]
    assert as_mapping(body["md_postprocess_job"], "md_postprocess_job")["job_id"] == (
        "plan-valid_ar_si_pr_hole-md-postprocess"
    )
    assert (output_dir / "lammps_execution_worker_bundle.json").exists()
    assert (output_dir / "md_postprocess_worker_bundle.json").exists()


def _lammps_execution_outputs(artifact_dir: str) -> tuple[str, ...]:
    return (
        f"{artifact_dir}/lammps_execution_result.json",
        f"{artifact_dir}/run_manifest.json",
        f"{artifact_dir}/surface_snapshot_before.data",
        f"{artifact_dir}/surface_snapshot_after.data",
        f"{artifact_dir}/incident.dump",
        f"{artifact_dir}/reflected.dump",
        f"{artifact_dir}/sputtered.dump",
        f"{artifact_dir}/implanted.dump",
        f"{artifact_dir}/traj.dump",
        f"{artifact_dir}/energy_depth_profile.csv",
        f"{artifact_dir}/damage_profile.csv",
        f"{artifact_dir}/roughness_rdf_descriptor.json",
        f"{artifact_dir}/log.lammps",
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
