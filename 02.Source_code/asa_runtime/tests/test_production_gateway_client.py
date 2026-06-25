from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def _mutable_request(name: str) -> dict[str, object]:
    return dict(_load_request(name))


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
    assert ledger["gateway_policy_id"] == "production-gateway-smoke-plan-v1"
    assert ledger["provider"] == "local_gateway"
    assert ledger["auth_mode"] == "gateway"
    assert sessions == {"orchestrator.jsonl", "research_agent.jsonl", "qa_agent.jsonl"}


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


def test_production_gateway_client_smoke_blocks_pathlike_agent_plan(tmp_path: Path) -> None:
    with _TraversalGateway() as gateway:
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

    assert result.returncode == 1
    assert "hard_blocker=gateway_agent_plan_invalid" in result.stdout
    assert ledger["hard_blockers"] == ["gateway_agent_plan_invalid"]
    assert not (tmp_path / "escape_specialist.jsonl").exists()
    assert not (tmp_path.parent / "escape_specialist.jsonl").exists()


def test_production_gateway_client_smoke_blocks_allowed_but_wrong_agent_plan(tmp_path: Path) -> None:
    with _WrongPlanGateway() as gateway:
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

    assert result.returncode == 1
    assert "hard_blocker=gateway_agent_plan_invalid" in result.stdout
    assert ledger["gateway_policy_id"] == "production-gateway-smoke-plan-v1"
    assert ledger["hard_blockers"] == ["gateway_agent_plan_invalid"]
    assert ledger["session_files"] == []


def _write_gateway_request(tmp_path: Path, base_url: str) -> Path:
    payload = _mutable_request("valid_ar_si_pr_hole.json")
    payload["model_provider"] = {
        "provider": "local_gateway",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "base_url": base_url,
        "auth_mode": "gateway",
        "api_key_env": "RUNTIME_GATEWAY_TOKEN",
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
    def log_message(self, format: str, *args: object) -> None:
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
                    "specialist": "research_agent",
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


class _TraversalGateway(_MockGateway):
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _TraversalGatewayHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)


class _WrongPlanGateway(_MockGateway):
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _WrongPlanGatewayHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)


class _TraversalGatewayHandler(_MockGatewayHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        self._write(
            {
                "gateway_request_id": "gw-python-smoke",
                "output_text": "production_gateway_client_ready",
                "agent_plan": {
                    "specialist": "../../escape_specialist",
                    "second_call": "qa_agent",
                },
            }
        )


class _WrongPlanGatewayHandler(_MockGatewayHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        self._write(
            {
                "gateway_request_id": "gw-python-smoke",
                "output_text": "production_gateway_client_ready",
                "agent_plan": {
                    "specialist": "md_agent",
                    "second_call": "qa_agent",
                },
            }
        )
