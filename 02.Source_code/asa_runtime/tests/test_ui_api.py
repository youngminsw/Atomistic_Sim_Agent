from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def test_ui_api_exposes_offline_fixture_runs_and_click_bundle() -> None:
    from sim_agent.ui import build_offline_fixture_request, build_ui_api_status, validate_ui_api_request

    status = build_ui_api_status()
    hole = validate_ui_api_request(build_offline_fixture_request("pr_hole_3d"))
    trench = validate_ui_api_request(build_offline_fixture_request("pr_trench_2d"))

    assert status.offline_fixtures == ("pr_hole_3d", "pr_trench_2d")
    assert "/api/run/offline" in status.route_paths
    assert "/api/agent/plan" in status.route_paths
    assert "/api/click-diagnostics" in status.route_paths
    assert "/api/knowledge/agent-context" in status.route_paths
    assert status.graphdb_database_name == "atomistic_sim_agent_knowledge"
    assert status.graphdb_write_requires_approval is True
    assert "openclaw" not in status.model_providers
    assert "oauth_gateway" not in status.model_providers
    assert "openai-codex" in status.model_providers
    assert "google-gemini-cli" in status.model_providers
    assert "gpt-5.5" in status.model_options
    assert {"api_key", "oauth", "gateway", "none"} <= set(status.auth_modes)
    assert "gateway_token" not in status.auth_modes
    assert "qa_agent" in status.agent_roles
    assert "production_gate" in status.agent_roles
    assert hole.can_run is True
    assert trench.can_run is True
    assert "--scene" in hole.runner_command
    assert "--image" in trench.runner_command


def test_ui_api_blocks_missing_iedf_before_run_command() -> None:
    from sim_agent.ui import build_offline_fixture_request, validate_ui_api_request

    request = build_offline_fixture_request("pr_hole_3d", iedf_ready=False)
    validation = validate_ui_api_request(request)

    assert validation.can_run is False
    assert validation.missing_fields == ("iedf",)
    assert validation.runner_command == ()


def test_start_ui_smoke_reports_static_root_and_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "start_ui.py"),
            "--offline-fixtures",
            "--port",
            "8765",
            "--smoke",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ui_smoke=true" in result.stdout
    assert "static_root=ui" in result.stdout
    assert "route=/api/run/offline" in result.stdout
    assert "fixture=pr_hole_3d" in result.stdout
    assert "fixture=pr_trench_2d" in result.stdout
    assert "auth_modes=api_key,oauth,gateway,none" in result.stdout
    assert "gateway_token" not in result.stdout


def test_ui_http_server_serves_status_and_blocks_missing_iedf() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        status_payload = as_mapping(
            json.loads(urlopen(f"http://{host}:{port}/api/status", timeout=5).read().decode("utf-8")),
            "status",
        )
        blocked_body, blocked_code = _post_json(
            f"http://{host}:{port}/api/run/offline",
            {
                "mode": "3d",
                "geometry_path": "tests/fixtures/scenes/pr_hole_scene.json",
                "kernel_path": "tests/fixtures/kernels/offline_ar_si_kernel.json",
                "events_path": "tests/fixtures/md_events/md_events_small.jsonl",
                "steps": 5,
                "ions": 8,
                "run_id": "http-hole",
                "compute_target": "gpu-5090",
                "iedf_ready": False,
                "iadf_ready": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "/api/run/offline" in status_payload["route_paths"]
    assert "/api/knowledge/agent-context" in status_payload["route_paths"]
    assert status_payload["graphdb"]["database_name"] == "atomistic_sim_agent_knowledge"
    assert status_payload["graphdb"]["write_requires_approval"] is True
    assert "openclaw" not in status_payload["model_providers"]
    assert "oauth_gateway" not in status_payload["model_providers"]
    assert "openai-codex" in status_payload["model_providers"]
    assert "gateway" in status_payload["auth_modes"]
    assert "gateway_token" not in status_payload["auth_modes"]
    assert "qa_agent" in status_payload["agent_roles"]
    assert "production_gate" in status_payload["agent_roles"]
    assert blocked_code == 400
    assert blocked_body["can_run"] is False
    assert blocked_body["missing_fields"] == ["iedf"]
    assert blocked_body["runner_command"] == []


def test_ui_http_server_denies_unauthorized_post_and_protects_non_loopback() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        status_payload = as_mapping(
            json.loads(urlopen(f"http://{host}:{port}/api/status", timeout=5).read().decode("utf-8")),
            "status",
        )
        body, status_code = _post_json(
            f"http://{host}:{port}/api/runtime/config",
            status_payload["runtime_config"],
            headers={"Content-Type": "application/json"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    security = as_mapping(status_payload["controller_security"], "controller_security")
    assert status_code == 401
    assert body["error"] == "unauthorized_state_change"
    assert security["token_exposed"] is False
    with pytest.raises(PermissionError, match="non_loopback_bind_requires_controller_token"):
        build_ui_http_server("0.0.0.0", 0, status.static_root)
    with pytest.raises(PermissionError, match="non_loopback_bind_requires_explicit_opt_in"):
        build_ui_http_server("0.0.0.0", 0, status.static_root, csrf_token="test-token")


def test_ui_http_server_serves_agent_graphdb_context() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = as_mapping(
            json.loads(urlopen(f"http://{host}:{port}/api/knowledge/agent-context", timeout=5).read().decode("utf-8")),
            "agent_graph_context",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert body["agent_access_enabled"] is True
    assert body["database_name"] == "atomistic_sim_agent_knowledge"
    assert body["write_requires_approval"] is True
    assert "password" not in body["connection"]
    assert {item["agent_id"] for item in body["role_queries"]} == {
        "orchestrator",
        "research_agent",
        "md_agent",
        "ml_agent",
        "feature_scale_agent",
        "qa_agent",
        "infra_agent",
    }
    assert any("force_field" in item["purpose"] for item in body["role_queries"])


def test_ui_http_server_rejects_unsafe_paths_and_oversized_body() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        absolute_body, absolute_code = _post_json(
            f"http://{host}:{port}/api/run/offline",
            {
                "mode": "3d",
                "geometry_path": str(SOURCE_ROOT / "tests" / "fixtures" / "scenes" / "pr_hole_scene.json"),
                "kernel_path": "tests/fixtures/kernels/offline_ar_si_kernel.json",
                "events_path": "tests/fixtures/md_events/md_events_small.jsonl",
                "steps": 2,
                "ions": 2,
                "run_id": "unsafe-absolute",
                "compute_target": "gpu-5090",
                "iedf_ready": True,
                "iadf_ready": True,
            },
        )
        parent_body, parent_code = _post_json(
            f"http://{host}:{port}/api/run/offline",
            {
                "mode": "3d",
                "geometry_path": "../outside.json",
                "kernel_path": "tests/fixtures/kernels/offline_ar_si_kernel.json",
                "events_path": "tests/fixtures/md_events/md_events_small.jsonl",
                "steps": 2,
                "ions": 2,
                "run_id": "unsafe-parent",
                "compute_target": "gpu-5090",
                "iedf_ready": True,
                "iadf_ready": True,
            },
        )
        large_body, large_code = _post_raw(
            f"http://{host}:{port}/api/run/offline",
            "x" * (300 * 1024),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert absolute_code == 400
    assert absolute_body["error"] == "absolute_input_paths_not_allowed"
    assert parent_code == 400
    assert parent_body["error"] == "parent_input_paths_not_allowed"
    assert large_code == 413
    assert large_body["error"] == "request_body_too_large"


def test_ui_http_server_agent_plan_reports_training_required_clarification() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body, status_code = _post_json(
            f"http://{host}:{port}/api/agent/plan",
            _load_request("ar_on_unknown_material.json"),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    trace = body["trace"]
    team = as_mapping(body["team_session_contract"], "team_session_contract")

    assert status_code == 200
    assert body["status"] == "clarification_required"
    assert team["heartbeat_interval_s"] == 3600
    assert team["qa_gates"]["slurm_job_script"] == "qa_before_submit"
    assert body["missing_fields"] == ["geometry", "material", "phase", "iedf", "iadf", "flux"]
    assert "model_training_required=true" in body["final_output"]
    assert "no_trained_expert_for_Ar_on_UnobtaniumFixture" in body["final_output"]
    assert [item["tool_name"] for item in trace] == [
        "inspect_request_inputs",
        "plan_simulation_input",
        "mark_surrogate_training_required",
    ]


def test_ui_http_server_agent_plan_rejects_bad_openclaw_base_url() -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body, status_code = _post_json(
            f"http://{host}:{port}/api/agent/plan",
            _load_request("openclaw_provider_bad_base_url.json"),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status_code == 400
    assert "ProviderConfigPolicyError" in body["error"]


def test_ui_http_server_executes_valid_offline_3d_run(tmp_path: Path) -> None:
    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.server import build_ui_http_server

    status = build_ui_api_status()
    server = build_ui_http_server("127.0.0.1", 0, status.static_root, csrf_token="test-token")
    host, port = server.server_address
    run_id = "pytest-http-hole-run"
    out_dir = SOURCE_ROOT / "evidence" / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body, status_code = _post_json(
            f"http://{host}:{port}/api/run/offline",
            {
                "mode": "3d",
                "geometry_path": "tests/fixtures/scenes/pr_hole_scene.json",
                "kernel_path": "tests/fixtures/kernels/offline_ar_si_kernel.json",
                "events_path": "tests/fixtures/md_events/md_events_small.jsonl",
                "steps": 3,
                "ions": 5,
                "run_id": run_id,
                "compute_target": "gpu-5090",
                "iedf_ready": True,
                "iadf_ready": True,
                "output_dir": f"evidence/{run_id}",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    manifest = as_mapping(json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")), "manifest")

    assert status_code == 200
    assert body["can_run"] is True
    assert body["run_status"] == "complete"
    assert body["manifest_path"] == f"evidence/{run_id}/manifest.json"
    assert body["click_index_path"] == f"evidence/{run_id}/click_index.json"
    assert body["qa_report_path"] == f"evidence/{run_id}/qa_report.json"
    assert body["qa_report"]["status"] == "pass"
    assert body["qa_report"]["evidence_scope"] == "offline_demo_fixture"
    assert body["qa_report"]["hard_blockers"] == []
    readiness = as_mapping(body["production_readiness"], "production_readiness")
    assert readiness["production_ready"] is False
    assert "model_endpoint_smoke_required" in readiness["hard_blockers"]
    assert "graphdb_live_ingest_required" in readiness["hard_blockers"]
    assert "production_feature_scale_report_required" in readiness["hard_blockers"]
    assert "approve_remote_or_long_compute_run" in readiness["user_actions"]
    assert body["artifact_links"]["profile_timeline"] == f"evidence/{run_id}/timeline.json"
    assert body["artifact_links"]["surrogate_model"] == f"evidence/{run_id}/empirical_mdn_model.json"
    assert body["artifact_links"]["qa_report"] == f"evidence/{run_id}/qa_report.json"
    assert {item["agent_id"] for item in body["agent_statuses"]} == {
        "orchestrator",
        "research_agent",
        "md_agent",
        "ml_agent",
        "feature_scale_agent",
        "qa_agent",
        "production_gate",
    }
    assert any(item["sender"] == "qa_agent" for item in body["agent_message_log"])
    assert any("qa_report_path=" in line for line in body["continuous_logs"])
    bundle = as_mapping(body["bundle"], "bundle")
    bundle_manifest = as_mapping(bundle["manifest"], "bundle.manifest")
    bundle_diagnostics = as_mapping(bundle["diagnostics"], "bundle.diagnostics")
    bundle_active_learning = as_mapping(bundle["active_learning_plan"], "bundle.active_learning_plan")
    bundle_qa = as_mapping(bundle["qa_report"], "bundle.qa_report")
    bundle_readiness = as_mapping(bundle["production_readiness"], "bundle.production_readiness")
    assert manifest["mode"] == "3d"
    assert manifest["run_status"] == "complete"
    assert bundle_manifest["feature_type"] == "hole"
    assert bundle_diagnostics["click_count"] > 1
    assert bundle_active_learning["controlled_event_probe_allowed"] is False
    assert bundle["surrogate_training_gate"]["accepted"] is True
    assert bundle["surrogate_model_manifest"]["quality_gate"]["decision"] == "accepted_for_feature_scale"
    assert bundle_qa["agent_id"] == "qa_agent"
    assert bundle_readiness["production_ready"] is False
    assert "uncertainty_map" in bundle
    assert (out_dir / "click_index.json").exists()
    shutil.rmtree(out_dir)


def test_ui_artifact_links_follow_runner_artifact_contract() -> None:
    from sim_agent.runner.artifact_contract import run_artifact_filename_map

    contract = run_artifact_filename_map()

    assert contract["manifest"] == "manifest.json"
    assert contract["profile_timeline"] == "timeline.json"
    assert contract["qa_report"] == "qa_report.json"
    assert set(contract) == {
        "manifest",
        "profile_timeline",
        "transport_field",
        "hit_history",
        "click_index",
        "uncertainty_map",
        "active_learning_plan",
        "surrogate_model",
        "surrogate_training_gate",
        "surrogate_model_manifest",
        "qa_report",
    }


def _post_json(url: str, payload: JsonMap, headers: dict[str, str] | None = None) -> tuple[JsonMap, int]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers or _auth_headers(),
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return as_mapping(json.loads(exc.read().decode("utf-8")), "response"), exc.code
    return as_mapping(json.loads(response.read().decode("utf-8")), "response"), response.status


def _post_raw(url: str, payload: str, headers: dict[str, str] | None = None) -> tuple[JsonMap, int]:
    request = Request(
        url,
        data=payload.encode("utf-8"),
        headers=headers or _auth_headers(),
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return as_mapping(json.loads(exc.read().decode("utf-8")), "response"), exc.code
    return as_mapping(json.loads(response.read().decode("utf-8")), "response"), response.status


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def _auth_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "X-ASA-CSRF-Token": "test-token"}
