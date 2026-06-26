from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from sim_agent.cli.workflow_live_llm_e2e import LIVE_WORKFLOW_IDS, WORKFLOW_ARGS_BEGIN, WORKFLOW_ARGS_END
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_provider


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_workflow_e2e_smoke_cli_drives_all_workflow_surfaces(tmp_path: Path) -> None:
    # Given: an output directory for a deterministic full-workflow e2e smoke.
    output_dir = tmp_path / "final-e2e"

    # When: the user-visible workflow e2e CLI is driven through the real command surface.
    result = _run_cli("--workflow-e2e-smoke", output_dir, "full-workflow-loop")

    # Then: stdout, ledger, transcript, and artifacts prove all workflow surfaces ran with one run id.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_e2e_smoke_status=succeeded" in result.stdout
    payload = json.loads((output_dir / "workflow-e2e.json").read_text(encoding="utf-8"))
    transcript = (output_dir / "workflow-transcript.txt").read_text(encoding="utf-8")
    run_id = payload["run_id"]

    assert payload["status"] == "succeeded"
    assert payload["workflow_ids"] == ["/deep-interview", "/ralplan", "/ultragoal", "/visual-qa", "/ultraresearch"]
    assert payload["skill_ids"] == ["insane-search"]
    assert all(row["status"] == "ready" for row in payload["workflow_results"])
    assert all(row["gate_status"] == "passed" for row in payload["workflow_results"])
    assert all(row["blockers"] == [] for row in payload["workflow_results"])
    assert payload["bounded_subagent_denials"] == {
        "/deep-interview": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ralplan": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ultragoal": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/visual-qa": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ultraresearch": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "insane-search": "persistent_skill_surface_unavailable_for_bounded_subagent",
    }
    assert f"run_id={run_id}" in transcript
    for command in payload["workflow_ids"]:
        assert command in transcript
    assert "skill=insane-search" in transcript
    for artifact in payload["artifacts"]:
        artifact_text = (output_dir / artifact).read_text(encoding="utf-8")
        assert run_id in artifact_text


def test_workflow_live_llm_e2e_cli_blocks_without_explicit_live_flag(tmp_path: Path) -> None:
    # Given: no ASA_LIVE_LLM_E2E opt-in in the environment.
    output_dir = tmp_path / "final-live-llm"

    # When: the live workflow e2e command is invoked.
    result = _run_cli("--workflow-live-llm-e2e", output_dir, "all-workflows")

    # Then: the command records typed provider-unavailable evidence and does not claim success.
    assert result.returncode == 1
    assert "workflow_live_llm_e2e_status=blocked" in result.stdout
    assert "workflow_live_llm_e2e_blocker=live_llm_opt_in_required" in result.stdout
    payload = json.loads((output_dir / "workflow-live-llm.json").read_text(encoding="utf-8"))
    provider_events = (output_dir / "provider-events.jsonl").read_text(encoding="utf-8")

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["live_llm_opt_in_required"]
    assert payload["live_llm"][0]["status"] == "blocked"
    assert payload["live_llm"][0]["blockers"] == ["live_llm_opt_in_required"]
    assert "live_llm_opt_in_required" in provider_events


def test_workflow_live_llm_e2e_cli_drives_provider_selected_workflow_start_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a stored openai-codex OAuth credential and a fake Codex backend.
    credential_store = tmp_path / "credentials.json"
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(credential_store))
    login_model_provider(
        {
            "provider": "openai-codex",
            "access_token": "stored-provider-token",
            "refresh_token": "stored-refresh-token",
            "auth_mode": "oauth",
            "expires_in_s": 3600,
        }
    )
    output_dir = tmp_path / "live-provider"

    with _WorkflowToolGateway(expected_token="stored-provider-token") as gateway:
        # When: the live workflow e2e command runs through the provider transport path.
        result = _run_cli(
            "--workflow-live-llm-e2e",
            output_dir,
            "provider-workflows",
            env_overrides={
                "ASA_LIVE_LLM_E2E": "1",
                "ASA_LIVE_LLM_E2E_BASE_URL": gateway.base_url,
                "ASA_LIVE_LLM_E2E_TIMEOUT_S": "5",
                CREDENTIAL_STORE_ENV: str(credential_store),
            },
        )

    # Then: every canonical workflow command was selected by the provider and executed by runtime tools.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_live_llm_e2e_status=succeeded" in result.stdout
    assert gateway.workflow_ids == list(LIVE_WORKFLOW_IDS)
    payload = json.loads((output_dir / "workflow-live-llm.json").read_text(encoding="utf-8"))
    provider_events = (output_dir / "provider-events.jsonl").read_text(encoding="utf-8")

    assert payload["status"] == "succeeded"
    assert payload["blockers"] == []
    assert payload["provider"]["provider"] == "openai-codex"
    assert payload["provider"]["model"] == "gpt-5.5"
    assert payload["provider"]["token_source"] == "credential_store"
    assert payload["workflow_ids"] == [f"/{workflow_id}" for workflow_id in LIVE_WORKFLOW_IDS]
    assert len(payload["live_llm"]) == len(LIVE_WORKFLOW_IDS)
    for row in payload["live_llm"]:
        assert row["status"] == "passed"
        assert row["selected_tools"] == ["workflow_start"]
        assert row["tool_results"][0]["tool_name"] == "workflow_start"
        expected_agent_status = "blocked" if row["workflow"] == "/deep-interview" else "succeeded"
        assert row["agent_loop_status"] == expected_agent_status
    assert "workflow_gate_response_required" in provider_events
    assert (output_dir / "sessions" / "visual-qa" / "workflows" / "visual-qa-live-screenshot.txt").is_file()


def _run_cli(
    flag: str,
    output_dir: Path,
    scenario: str,
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("ASA_LIVE_LLM_E2E", None)
    env.pop("ASA_LIVE_LLM_E2E_BASE_URL", None)
    env.pop("ASA_LIVE_LLM_E2E_TIMEOUT_S", None)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            flag,
            "--scenario",
            scenario,
            "--output-dir",
            str(output_dir),
        ],
        cwd=SOURCE_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )


class _WorkflowToolGateway:
    def __init__(self, *, expected_token: str) -> None:
        self._handler = type(
            "_WorkflowToolGatewayHandler",
            (_WorkflowToolGatewayHandler,),
            {
                "expected_token": expected_token,
                "workflow_ids": [],
            },
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def workflow_ids(self) -> list[str]:
        return list(self._handler.workflow_ids)

    def __enter__(self) -> "_WorkflowToolGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _WorkflowToolGatewayHandler(BaseHTTPRequestHandler):
    expected_token = "stored-provider-token"
    workflow_ids: list[str] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/codex/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        if self.headers.get("authorization") != f"Bearer {self.expected_token}":
            self._write({"error": {"code": "missing_gateway_credentials"}}, status=401)
            return
        length = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        arguments = _workflow_arguments_from_provider_payload(body)
        workflow_id = arguments["workflow_id"]
        assert isinstance(workflow_id, str)
        self.__class__.workflow_ids.append(workflow_id)
        self._write(
            {
                "output": [
                    {
                        "type": "tool_call",
                        "name": "workflow_start",
                        "arguments": arguments,
                    }
                ]
            }
        )

    def _write(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _workflow_arguments_from_provider_payload(payload: dict[str, object]) -> dict[str, object]:
    inputs = payload["input"]
    assert isinstance(inputs, list) and inputs
    first = inputs[0]
    assert isinstance(first, dict)
    content = first["content"]
    assert isinstance(content, str)
    raw = content.split(WORKFLOW_ARGS_BEGIN, 1)[1].split(WORKFLOW_ARGS_END, 1)[0].strip()
    decoded = json.loads(raw)
    assert isinstance(decoded, dict)
    return decoded
