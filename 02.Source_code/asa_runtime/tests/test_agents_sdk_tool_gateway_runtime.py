from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_fake_local_gateway_dispatches_attached_runtime_tools_and_sessions(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_tool_gateway_runtime, write_tool_gateway_runtime_ledger
    from sim_agent.llm_endpoints import ModelProviderConfig

    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "local_gateway",
            "model": "gpt-5.3",
            "reasoning_effort": "high",
            "base_url": "http://local-gateway.test/v1",
            "auth_mode": "none",
        }
    )

    result = run_agents_sdk_tool_gateway_runtime(
        {
            "request_id": "tool-gateway",
            "user_goal": "Prove attached runtime tools execute",
        },
        endpoint,
        tmp_path,
    )
    ledger_path = write_tool_gateway_runtime_ledger(tmp_path, result)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert result.status == "succeeded"
    assert result.gateway_policy_id == "local-tool-gateway-smoke-v1"
    assert result.gateway_mode == "local_smoke"
    assert result.gateway_request_id == "fake-local-tool-gateway"
    assert result.tool_results[0].tool_name == "bash_process"
    assert result.tool_results[0].output["stdout"] == "gateway-tool-ok\n"
    assert result.tool_results[1].tool_name == "graphdb_dry_run"
    assert result.tool_results[1].output["neo4j_write_enabled"] is False
    assert (tmp_path / result.tool_results[0].artifact_ref).is_file()
    assert {Path(path).name for path in result.session_files} == {"orchestrator.jsonl", "tool_runtime.jsonl"}
    assert ledger["attached_tools"] == [
        "bash_process",
        "artifact_write",
        "graphdb_dry_run",
        "agent_message",
        "handoff_task",
        "subagent_task",
        "subagent_inspect",
        "subagent_control",
        "skill_invoke",
        "workflow_start",
    ]
    assert ledger["gateway_policy_id"] == "local-tool-gateway-smoke-v1"
    assert ledger["gateway_mode"] == "local_smoke"
    assert ledger["status"] == "succeeded"
    assert ledger["tool_results"][0]["status"] == "succeeded"
    assert ledger["session_files"] == list(result.session_files)
    assert ledger["agent_loop_status"] == "succeeded"
    assert ledger["agent_session_agent_id"] == "orchestrator"
    assert ledger["model_selected_tools"] == ["bash_process", "graphdb_dry_run"]
    assert {schema["name"] for schema in ledger["model_visible_tools"]} >= {
        "bash_process",
        "artifact_write",
        "graphdb_dry_run",
        "validate_simulation_request",
    }
    assert {event["event_type"] for event in ledger["loop_trace"]} >= {
        "asa_agent_session_created",
        "model_visible_tools_registered",
        "model_tool_selected",
        "tool_executed",
        "agent_loop_completed",
    }


def test_tool_gateway_runtime_fails_closed_without_gateway_credentials(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_tool_gateway_runtime
    from sim_agent.llm_endpoints import ModelProviderConfig

    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "oauth_gateway",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://gateway.example/v1",
            "auth_mode": "gateway",
            "api_key_env": "ASA_TEST_MISSING_GATEWAY_TOKEN",
        }
    )

    result = run_agents_sdk_tool_gateway_runtime(
        {
            "request_id": "missing-credentials",
            "user_goal": "This must not silently fall back",
        },
        endpoint,
        tmp_path,
    )

    assert result.status == "blocked"
    assert result.blockers == ("missing_gateway_credentials",)
    assert result.tool_results == ()
    assert result.final_output == "blocked"


def test_tui_runtime_tool_gateway_dispatches_tools_without_long_cli_options(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input=(
            "/model set --provider local_gateway --model gpt-5.3 "
            "--base-url http://local-gateway.test/v1 --auth-mode none\n"
            "/runtime tools\n"
            "/exit\n"
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime_tool_gateway=true" in result.stdout
    assert "runtime_status=succeeded" in result.stdout
    assert "gateway_policy_id=local-tool-gateway-smoke-v1" in result.stdout
    assert "gateway_mode=local_smoke" in result.stdout
    assert "gateway_request_id=fake-local-tool-gateway" in result.stdout
    assert "tool_result=bash_process:succeeded" in result.stdout
    assert "tool_result=graphdb_dry_run:succeeded" in result.stdout


def test_tui_runtime_tool_gateway_uses_stored_oauth_token(tmp_path: Path, monkeypatch) -> None:
    from io import StringIO

    from sim_agent.agents_sdk_runtime.tool_gateway_runtime import ToolGatewayRuntimeResult
    from sim_agent.cli import tui_runtime
    from sim_agent.cli.tui_runtime import handle_runtime
    from sim_agent.cli.tui_state import initial_state
    from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_gateway

    captured: dict[str, str | None] = {}
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(tmp_path / "credentials.json"))
    login_model_gateway(
        {
            "provider": "openai-codex",
            "access_token": "stored-oauth-token",
            "refresh_token": "stored-refresh-token",
            "auth_mode": "oauth",
            "expires_in_s": 3600,
        }
    )

    def fake_run_agents_sdk_tool_gateway_runtime(
        _payload: dict[str, object],
        _endpoint: object,
        _output_dir: Path,
        *,
        api_key: str | None = None,
    ) -> ToolGatewayRuntimeResult:
        captured["api_key"] = api_key
        return ToolGatewayRuntimeResult(
            run_id="fake-run",
            session_id="fake-session",
            status="blocked",
            provider="openai-codex",
            model="gpt-5-codex",
            auth_mode="gateway",
            gateway_policy_id="fake-policy",
            gateway_mode="fake-mode",
            gateway_request_id=None,
            attached_tools=(),
            tool_results=(),
            session_files=(),
            blockers=("live_gateway_tool_dispatch_not_configured",),
            final_output="blocked",
        )

    monkeypatch.setattr(
        tui_runtime,
        "run_agents_sdk_tool_gateway_runtime",
        fake_run_agents_sdk_tool_gateway_runtime,
    )

    output = StringIO()
    handle_runtime(("tools",), initial_state(tmp_path / "session"), output)

    assert captured["api_key"] == "stored-oauth-token"
    assert "runtime_tool_gateway=true" in output.getvalue()
