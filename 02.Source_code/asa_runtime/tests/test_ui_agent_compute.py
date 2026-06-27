from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_ui_http_prepares_md_campaign_worker_bundle(tmp_path: Path) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    output_dir = tmp_path / "agent-plan"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body, status_code = _post_json(
            f"http://{host}:{port}/api/agent/prepare-md-campaign-worker-bundle",
            {
                "request": _amorphous_request("valid_ar_si_pr_hole.json"),
                "output_dir": str(output_dir),
                "host": "gpu-5090",
                "environment_name": "atomistic-sim-gpu",
                "remote_user": "swym",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    job = as_mapping(body["job"], "job")
    worker = as_mapping(body["worker_bundle"], "worker_bundle")
    assert status_code == 200
    assert body["prepared"] is True
    assert body["artifact_dir"] == str(output_dir)
    assert body["amorphous_structure_prep_manifest_path"] == str(
        output_dir / "amorphous_structure_prep" / "amorphous_structure_prep_manifest.json"
    )
    prep_job = as_mapping(body["amorphous_structure_prep_job"], "prep_job")
    prep_worker = as_mapping(body["amorphous_structure_prep_worker_bundle"], "prep_worker")
    assert body["amorphous_structure_prep_job_path"] == str(
        output_dir / "amorphous_structure_prep_job.json"
    )
    assert body["amorphous_structure_prep_worker_path"] == str(
        output_dir / "amorphous_structure_prep_worker_bundle.json"
    )
    assert prep_job["job_id"] == "plan-valid_ar_si_pr_hole-amorphous-structure-prep"
    assert prep_job["requires_cuda"] is False
    assert prep_worker["host_alias"] == "gpu-5090"
    assert prep_worker["output_paths"] == prep_job["outputs"]
    prep_requirements = as_mapping(
        prep_worker["capability_requirements"],
        "prep_capability_requirements",
    )
    assert prep_requirements["requires_lammps"] is True
    assert job["job_id"] == "plan-valid_ar_si_pr_hole-md-campaign"
    assert job["outputs"] == [
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_contract.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/incident_schedule.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_state.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/in.atomistic_campaign",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_input_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/surface_snapshot_before.data",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/Si.tersoff",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_assets_manifest.json",
        "artifacts/plan-valid_ar_si_pr_hole-md-campaign/lammps_execution_plan.json",
    ]
    assert worker["host_alias"] == "gpu-5090"
    assert worker["output_paths"] == job["outputs"]
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "md_campaign_job.json").exists()
    assert (output_dir / "worker_bundle.json").exists()
    assert (output_dir / "amorphous_structure_prep_job.json").exists()
    assert (output_dir / "amorphous_structure_prep_worker_bundle.json").exists()
    assert (
        output_dir / "amorphous_structure_prep" / "amorphous_structure_source.json"
    ).exists()


def test_ui_http_prepares_remote_execution_plan_when_ssh_target_is_present(tmp_path: Path) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    output_dir = tmp_path / "agent-plan"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body, status_code = _post_json(
            f"http://{host}:{port}/api/agent/prepare-md-campaign-worker-bundle",
            {
                "request": _amorphous_request("valid_ar_si_pr_hole.json"),
                "output_dir": str(output_dir),
                "host": "gpu-5090",
                "environment_name": "atomistic-sim-gpu",
                "remote_user": "swym",
                "ssh_target": "swym@10.24.12.85",
                "ssh_port": 55555,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    remote_plan = as_mapping(body["remote_execution_plan"], "remote_execution_plan")
    prep_remote_plan = as_mapping(
        body["amorphous_structure_prep_remote_execution_plan"],
        "prep_remote_execution_plan",
    )
    assert status_code == 200
    assert body["remote_plan_path"] == str(output_dir / "remote" / "remote_plan.json")
    assert body["amorphous_structure_prep_remote_plan_path"] == str(
        output_dir / "remote" / "amorphous_structure_prep_remote_plan.json"
    )
    assert remote_plan["ssh_target"] == "swym@10.24.12.85"
    assert remote_plan["ssh_port"] == 55555
    assert remote_plan["kind"] == "remote_execution_plan"
    assert remote_plan["created_by"] == "asa_runtime"
    assert remote_plan["output_root"] == str(output_dir.resolve())
    assert "plan_sha256" in remote_plan
    assert prep_remote_plan["ssh_target"] == "swym@10.24.12.85"
    assert "prepare_amorphous_structure_job.py" in prep_remote_plan["execution_command"]
    assert str(remote_plan["execution_command"]).startswith("ssh -p 55555 swym@10.24.12.85 ")
    assert (output_dir / "remote" / "remote_plan.json").exists()
    assert (output_dir / "remote" / "amorphous_structure_prep_remote_plan.json").exists()


def _post_json(url: str, payload: JsonMap) -> tuple[JsonMap, int]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-ASA-CSRF-Token": "test-token"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return as_mapping(json.loads(exc.read().decode("utf-8")), "response"), exc.code
    return as_mapping(json.loads(response.read().decode("utf-8")), "response"), response.status


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def _amorphous_request(name: str) -> JsonMap:
    request = dict(_load_request(name))
    scene = dict(as_mapping(request["scene"], "scene"))
    surface = dict(as_mapping(scene["surface_state"], "surface_state"))
    surface["phase"] = "amorphous"
    surface["md_box"] = {
        "atom_count": 5000,
        "expected_cascade_depth_nm": 6.0,
        "fixed_depth_nm": 1.0,
        "lateral_x_nm": 12.0,
        "lateral_y_nm": 12.0,
        "mobile_depth_nm": 9.0,
        "run_length_ps": 2.0,
        "thermostat_depth_nm": 1.0,
        "timestep_fs": 0.1,
    }
    scene["surface_state"] = surface
    request["scene"] = scene
    return request
