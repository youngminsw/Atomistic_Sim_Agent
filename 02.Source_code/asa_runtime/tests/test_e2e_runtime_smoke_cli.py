from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import TracebackType

from sim_agent.runtime_config import (
    ActiveModelProfileRuntimeConfig,
    ModelEndpointRuntimeConfig,
    default_runtime_config,
    save_runtime_config,
)
from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_e2e_runtime_smoke_cli_writes_redacted_live_turn_evidence_with_test_bypass(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        config_path = _write_runtime_config(tmp_path / "runtime-config.json", gateway.base_url)
        output_json = tmp_path / "evidence" / "smoke.json"

        result = _run_smoke(output_json, config_path, allow_hardgate_bypass=True)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "e2e_runtime_smoke=true" in result.stdout
    assert gateway.authorizations == ["Bearer fake-e2e-token"] * 3
    assert [body["model"] for body in gateway.request_bodies] == ["gpt-5.5"] * 3
    assert _tool_names(gateway.request_bodies[0]) >= {"subagent_task", "handoff_task"}
    assert _tool_names(gateway.request_bodies[1]) == {"artifact_write", "subagent_inspect"}
    assert _tool_names(gateway.request_bodies[2]) >= {"artifact_write", "subagent_task"}
    assert payload["status"] == "succeeded"
    assert payload["blockers"] == []
    assert payload["model_profile"] == "codex-pro"
    assert payload["model"] == {
        "provider": "openai-codex",
        "name": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "base_url": gateway.base_url,
        "auth_mode": "gateway",
        "api_key_env": "ASA_E2E_GATEWAY_TOKEN",
    }
    assert payload["hardgate_bypass_mode"] == "test_only"
    assert payload["selected_tools"] == ["subagent_task"]
    assert payload["subagent_task_selected"] is True
    assert payload["subagent_runs"][0]["subagent_id"] == "e2e-planner-smoke"
    assert payload["subagent_runs"][0]["selected_tools"] == ["artifact_write"]
    assert payload["message_bus"]["status"] == "succeeded"
    assert payload["handoff"]["status"] == "live_completed"
    assert payload["handoff"]["target_agent"] == "md_agent"
    assert payload["compaction"]["compact_status"] == "compacted"
    assert payload["compaction_replay"]["compact_status"] == "replayed"
    assert payload["resume"]["opened_as"] == "resumed"
    assert payload["timeline"]["event_count"] > 0
    assert payload["ledger_reconciliation"]["subagent_run_count"] == 1
    assert payload["destructive_write_statement"] == "No MD, remote execution, or GraphDB destructive write ran."
    assert payload["destructive_writes_ran"] == {"graphdb": False, "md": False, "remote": False}
    assert Path(payload["global_session_path"]).is_file()
    assert Path(payload["agent_session_dir"]).is_dir()
    assert "fake-e2e-token" not in output_json.read_text(encoding="utf-8")


def test_e2e_runtime_smoke_cli_omits_test_bypass_without_flag(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        config_path = _write_runtime_config(tmp_path / "runtime-config.json", gateway.base_url)
        output_json = tmp_path / "evidence" / "smoke-no-bypass.json"

        result = _run_smoke(output_json, config_path, allow_hardgate_bypass=False)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert gateway.call_count == 3
    assert "hardgate_bypass_mode" not in payload


def _write_runtime_config(path: Path, base_url: str) -> Path:
    default = default_runtime_config()
    config = replace(
        default,
        evidence_root=str(path.parent / "runtime-evidence"),
        model_endpoint=ModelEndpointRuntimeConfig(
            provider="openai-codex",
            model="gpt-5.5",
            reasoning_effort="xhigh",
            base_url=base_url,
            auth_mode="gateway",
            api_key_env="ASA_E2E_GATEWAY_TOKEN",
        ),
        active_profile=ActiveModelProfileRuntimeConfig(name="codex-pro", customized=False),
    )
    return save_runtime_config(config, path)


def _run_smoke(
    output_json: Path,
    config_path: Path,
    *,
    allow_hardgate_bypass: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    env["ASA_E2E_GATEWAY_TOKEN"] = "fake-e2e-token"
    command = [
        sys.executable,
        "-m",
        "sim_agent.cli.main",
        "--e2e-runtime-smoke",
        "--model-profile",
        "codex-pro",
        "--scenario",
        "orchestrator_subagent_tool_loop",
        "--output-json",
        str(output_json),
    ]
    if allow_hardgate_bypass:
        command.insert(-2, "--allow-hardgate-bypass")
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )


class _LiveTurnGateway:
    def __init__(self) -> None:
        self._handler = type(
            "_E2ELiveTurnGatewayHandler",
            (_LiveTurnGatewayHandler,),
            {"request_bodies": [], "authorizations": []},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def request_bodies(self) -> list[JsonMap]:
        return list(self._handler.request_bodies)

    @property
    def authorizations(self) -> list[str]:
        return list(self._handler.authorizations)

    @property
    def call_count(self) -> int:
        return len(self._handler.request_bodies)

    def __enter__(self) -> "_LiveTurnGateway":
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _LiveTurnGatewayHandler(BaseHTTPRequestHandler):
    request_bodies: list[JsonMap] = []
    authorizations: list[str] = []

    def log_message(self, format: str, *args: str) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/codex/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        length = int(self.headers["content-length"])
        self.__class__.authorizations.append(self.headers.get("authorization", ""))
        self.__class__.request_bodies.append(json.loads(self.rfile.read(length).decode("utf-8")))
        self._write(_gateway_response(len(self.__class__.request_bodies)))

    def _write(self, payload: JsonMap, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _gateway_response(call_number: int) -> JsonMap:
    if call_number == 1:
        return _tool_response(
            "subagent_task",
            {
                "caller_agent": "orchestrator",
                "preset": "planner",
                "task_id": "e2e-planner-smoke",
                "task": "Use artifact_write to write subagent_report.md with e2e subagent child ok.",
                "depth": 1,
            },
        )
    if call_number == 2:
        return _tool_response("artifact_write", {"relative_path": "subagent_report.md", "content": "e2e subagent child ok"})
    return _tool_response(
        "artifact_write",
        {"relative_path": "e2e_runtime_smoke/handoff_report.md", "content": "e2e handoff target ok"},
    )


def _tool_response(name: str, arguments: JsonMap) -> JsonMap:
    return {"output": [{"type": "function_call", "name": name, "arguments": arguments}]}


def _tool_names(body: JsonMap) -> set[str]:
    tools = body.get("tools")
    assert isinstance(tools, list)
    return {tool["name"] for tool in tools if isinstance(tool, dict)}
