from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def test_agents_sdk_runtime_dry_run_records_all_agent_handoffs_and_approvals(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run, write_agents_sdk_runtime_ledger
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    payload["host"] = "gpu-5090"
    payload["estimated_runtime_s"] = 3700
    payload["graphdb"] = {"mode": "attempt_write"}
    endpoint = ModelProviderConfig.from_mapping(as_mapping(payload["llm_endpoint"], "llm_endpoint"))

    result = run_agents_sdk_runtime_dry_run(payload, endpoint)
    ledger_path = write_agents_sdk_runtime_ledger(tmp_path, result)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert result.handoff_sequence == (
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "research_graphdb_agent",
        "qa_agent",
    )
    assert len(result.messages) == 10
    assert ledger["session_id"] == "sim-agent-sdk-valid_ar_si_pr_hole"
    assert ledger["provider"] == "openclaw"
    assert ledger["model"] == "gpt-5.5"
    assert ledger["auth_mode"] == "oauth"
    assert {gate["gate_id"]: gate["status"] for gate in ledger["approval_gates"]} == {
        "remote_execution": "required",
        "long_runtime": "required",
        "graphdb_write": "required",
        "destructive_action": "not_required",
    }
    assert "handoff_to_research_graphdb_agent:research_graphdb_agent" in {
        item["summary"] for item in ledger["trace"]
    }


def test_agents_sdk_runtime_records_approved_boundaries() -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    payload["remote_execution"] = True
    payload["destructive_action"] = True
    payload["approvals"] = {
        "remote_execution": True,
        "destructive_action": True,
    }
    endpoint = ModelProviderConfig.from_mapping(as_mapping(payload["llm_endpoint"], "llm_endpoint"))

    result = run_agents_sdk_runtime_dry_run(payload, endpoint)

    assert {gate.gate_id: gate.status.value for gate in result.approval_gates}["remote_execution"] == "approved"
    assert {gate.gate_id: gate.status.value for gate in result.approval_gates}["destructive_action"] == "approved"


def test_actual_openai_agents_sdk_team_and_fake_gateway_smoke() -> None:
    pytest.importorskip("agents")

    from sim_agent.agents_sdk_runtime import build_agents_sdk_team, run_agents_sdk_runtime_dry_run
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    endpoint = ModelProviderConfig.from_mapping(as_mapping(payload["llm_endpoint"], "llm_endpoint"))
    team = build_agents_sdk_team(endpoint, "sdk-smoke")
    result = run_agents_sdk_runtime_dry_run(payload, endpoint, run_sdk_smoke=True)

    assert team.orchestrator.name == "Orchestrator"
    assert set(team.specialists) == {
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "research_graphdb_agent",
        "qa_agent",
    }
    assert "handoff_to_md_agent" in team.handoff_tool_names
    assert result.sdk_available is True
    assert result.sdk_run_completed is True
    assert result.final_output == "agents_sdk_runtime_ready"


def test_smoke_agents_sdk_runtime_cli_writes_ledger(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agents_sdk_runtime.py"),
            "--request",
            str(REQUEST_ROOT / "valid_ar_si_pr_hole.json"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((tmp_path / "agents_sdk_runtime_ledger.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agents_sdk_runtime_ledger_path=" in result.stdout
    assert ledger["run_id"] == "agents-sdk-valid_ar_si_pr_hole"
    assert ledger["handoff_sequence"] == [
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "research_graphdb_agent",
        "qa_agent",
    ]


def test_agent_team_session_smoke_writes_durable_sessions_and_call_matrix(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agents_sdk_runtime.py"),
            "--request",
            str(REQUEST_ROOT / "valid_ar_si_pr_hole.json"),
            "--output-dir",
            str(tmp_path),
            "--team-session-smoke",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((tmp_path / "agent_team_session_ledger.json").read_text(encoding="utf-8"))
    session_files = {Path(path).name for path in ledger["session_files"]}

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_team_session_ledger_path=" in result.stdout
    assert "deadlock=false" in result.stdout
    assert ledger["status"] == "ready"
    assert ledger["heartbeat_interval_s"] == 3600
    assert session_files == {
        "orchestrator.jsonl",
        "md_agent.jsonl",
        "ml_mdn_agent.jsonl",
        "feature_scale_agent.jsonl",
        "research_graphdb_agent.jsonl",
        "qa_agent.jsonl",
    }
    assert set(ledger["call_matrix"]["orchestrator"]) == {
        "md_agent",
        "ml_mdn_agent",
        "feature_scale_agent",
        "research_graphdb_agent",
        "qa_agent",
    }
    for role_id in ("md_agent", "ml_mdn_agent", "feature_scale_agent"):
        assert set(ledger["call_matrix"][role_id]) == {
            "orchestrator",
            "research_graphdb_agent",
            "qa_agent",
        }
    assert any(event["event_type"] == "heartbeat_registered" for event in ledger["events"])
    assert any(event["event_type"] == "context_compaction_checkpoint" for event in ledger["events"])


def test_agent_team_session_smoke_recovers_failed_peer_without_deadlock(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agents_sdk_runtime.py"),
            "--request",
            str(REQUEST_ROOT / "valid_ar_si_pr_hole.json"),
            "--output-dir",
            str(tmp_path),
            "--team-session-smoke",
            "--simulate-agent-failure",
            "md_agent",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((tmp_path / "agent_team_session_ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "deadlock=false" in result.stdout
    assert "agent_failure=md_agent" in result.stdout
    assert ledger["status"] == "degraded"
    assert ledger["deadlock"] is False
    assert "agent_failure:md_agent" in ledger["recoverable_events"]
    assert any(
        event["event_type"] == "recovery_route" and event["agent_id"] == "orchestrator"
        for event in ledger["events"]
    )


def test_agent_team_session_smoke_blocks_slurm_job_script_without_qa_gate(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agents_sdk_runtime.py"),
            "--request",
            str(REQUEST_ROOT / "valid_ar_si_pr_hole.json"),
            "--output-dir",
            str(tmp_path),
            "--team-session-smoke",
            "--slurm-job-script",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((tmp_path / "agent_team_session_ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert "hard_blocker=qa_job_script_review_required" in result.stdout
    assert ledger["status"] == "blocked"
    assert ledger["qa_gates"]["slurm_job_script"] == "required"
    assert ledger["hard_blockers"] == ["qa_job_script_review_required"]


def test_production_gateway_client_smoke_records_request_and_sessions(tmp_path: Path) -> None:
    with _MockGateway() as gateway:
        request_path = _write_gateway_request(tmp_path, gateway.base_url)
        result = subprocess.run(
            [
                sys.executable,
                str(SOURCE_ROOT / "scripts" / "smoke_production_gateway_client.py"),
                "--request",
                str(request_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--api-key",
                "test-gateway-token",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    ledger = json.loads((tmp_path / "out" / "production_gateway_smoke_ledger.json").read_text(encoding="utf-8"))
    sessions = {Path(path).name for path in ledger["session_files"]}

    assert result.returncode == 0, result.stdout + result.stderr
    assert "production_smoke=true" in result.stdout
    assert "fake_gateway_model=false" in result.stdout
    assert "gateway_request_id=gw-python-smoke" in result.stdout
    assert ledger["production_smoke"] is True
    assert ledger["fake_gateway_model"] is False
    assert ledger["gateway_request_id"] == "gw-python-smoke"
    assert ledger["provider"] == "oauth_gateway"
    assert ledger["auth_mode"] == "gateway"
    assert sessions == {"orchestrator.jsonl", "research_graphdb_agent.jsonl", "qa_agent.jsonl"}


def test_production_gateway_client_smoke_records_missing_credential_blocker(tmp_path: Path) -> None:
    request_path = _write_gateway_request(tmp_path, "http://127.0.0.1:9/v1")
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_production_gateway_client.py"),
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )

    ledger = json.loads((tmp_path / "out" / "production_gateway_smoke_ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert "hard_blocker=missing_gateway_credentials" in result.stdout
    assert ledger["hard_blockers"] == ["missing_gateway_credentials"]
    assert ledger["offline"] is False


def test_production_gateway_client_smoke_refuses_offline(tmp_path: Path) -> None:
    request_path = _write_gateway_request(tmp_path, "http://127.0.0.1:9/v1")
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_production_gateway_client.py"),
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--offline",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((tmp_path / "out" / "production_gateway_smoke_ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert "hard_blocker=production_smoke_refuses_offline" in result.stdout
    assert ledger["offline"] is True


def _write_gateway_request(tmp_path: Path, base_url: str) -> Path:
    payload = _load_request("valid_ar_si_pr_hole.json")
    payload["llm_endpoint"] = {
        "provider": "oauth_gateway",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "base_url": base_url,
        "auth_mode": "gateway",
        "api_key_env": "MODEL_GATEWAY_TOKEN",
    }
    path = tmp_path / "gateway_request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _MockGateway:
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _MockGatewayHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "_MockGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _MockGatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write({"ok": True, "gateway_request_id": "gw-health"})
            return
        if self.path == "/v1/models":
            self._write({"object": "list", "data": [{"id": "gpt-5.5"}], "gateway_request_id": "gw-models"})
            return
        self._write({"error": {"code": "not_found"}}, status=404)

    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        if self.headers.get("authorization") != "Bearer test-gateway-token":
            self._write({"error": {"code": "missing_gateway_credentials"}}, status=401)
            return
        self._write(
            {
                "gateway_request_id": "gw-python-smoke",
                "output_text": "production_gateway_client_ready",
                "agent_plan": {
                    "specialist": "research_graphdb_agent",
                    "second_call": "qa_agent",
                },
            }
        )

    def _write(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
