from __future__ import annotations

import base64
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession, ModelToolChoiceBlocked
from sim_agent.agents_sdk_runtime.gateway_client_types import GatewayClientSmokeError
from sim_agent.agents_sdk_runtime.prompt_assets import load_domain_role_prompt
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.ui.model_auth import CREDENTIAL_STORE_ENV, login_model_provider


def test_provider_tool_choice_model_posts_visible_tool_schemas_and_parses_output_tool_call(tmp_path: Path) -> None:
    with _ToolChoiceGateway(_tool_call_response("artifact_write", {"relative_path": "provider/evidence.txt", "content": "ok"})) as gateway:
        session = _session(tmp_path, gateway.base_url)
        model = ProviderToolChoiceModel(api_key="test-token")

        result = AgentLoop(session, model).run()

    body = gateway.request_body
    tool_names = {tool["name"] for tool in body["tools"]}
    assert result.status == "succeeded"
    assert result.model_id == "oauth_gateway/gpt-5.5"
    assert tuple(call.tool_name for call in result.selected_tools) == ("artifact_write",)
    assert (tmp_path / "artifacts" / "provider" / "evidence.txt").read_text(encoding="utf-8") == "ok"
    assert body["model"] == "gpt-5.5"
    assert body["input"] == [{"role": "user", "content": "Write provider-selected evidence"}]
    assert isinstance(body["instructions"], str)
    assert "ASA common system policy" in body["instructions"]
    assert "You are ASA" in body["instructions"]
    assert "Workflow policy" in body["instructions"]
    assert "Request gate" in body["instructions"]
    assert "Domain agent role" in body["instructions"]
    assert "You are the Orchestrator" in body["instructions"]
    assert "artifact_write" in tool_names
    assert body["reasoning"]["effort"] == "high"
    assert body["metadata"]["agent_id"] == "orchestrator"
    assert body["metadata"]["session_id"] == "provider-session"
    manifest = json.loads((tmp_path / "prompt_assembly_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "asa_prompt_assembly_manifest_v1"
    assert manifest["agent_id"] == "orchestrator"
    assert manifest["api_protocol"] == "openai_responses"
    assert manifest["layer_kinds"][:3] == ["system_policy", "workflow_policy", "domain_role"]
    assert manifest["messages"] == [{"role": "user", "content": "Write provider-selected evidence"}]
    assert "artifact_write" in manifest["tool_names"]


def test_provider_tool_choice_model_parses_tool_calls_list_shape(tmp_path: Path) -> None:
    with _ToolChoiceGateway({"tool_calls": [{"name": "graphdb_dry_run", "arguments": {"database_name": "asa"}}]}) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "succeeded"
    assert tuple(call.tool_name for call in result.selected_tools) == ("graphdb_dry_run",)


def test_provider_tool_choice_model_uses_stored_credential_token_when_no_explicit_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MODEL_GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv(CREDENTIAL_STORE_ENV, str(tmp_path / "credentials.json"))
    login_model_provider(
        {
            "provider": "oauth_gateway",
            "access_token": "stored-provider-token",
            "refresh_token": "stored-refresh-token",
            "auth_mode": "oauth",
            "expires_in_s": 3600,
        }
    )
    with _ToolChoiceGateway(
        _tool_call_response("artifact_write", {"relative_path": "provider/stored-token.txt", "content": "ok"}),
        expected_token="stored-provider-token",
    ) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url, auth_mode="oauth"), ProviderToolChoiceModel()).run()

    assert result.status == "succeeded"
    assert (tmp_path / "artifacts" / "provider" / "stored-token.txt").read_text(encoding="utf-8") == "ok"


def test_provider_tool_choice_model_adds_openai_codex_backend_headers(tmp_path: Path) -> None:
    token = _fake_codex_token("acct-test-123")
    with _ToolChoiceGateway(
        _tool_call_response("artifact_write", {"relative_path": "provider/codex.txt", "content": "ok"}),
        expected_path="/codex/responses",
        expected_token=token,
    ) as gateway:
        session = _session(tmp_path, gateway.backend_base_url, provider="openai-codex", auth_mode="oauth")
        result = AgentLoop(session, ProviderToolChoiceModel(api_key=token)).run()

    assert result.status == "succeeded"
    assert gateway.request_headers["openai_beta"] == "responses=experimental"
    assert gateway.request_headers["originator"] == "pi"
    assert gateway.request_headers["account_id"] == "acct-test-123"


def test_provider_tool_choice_model_retries_endpoint_unreachable_before_blocking(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    calls: list[dict[str, object]] = []

    def flaky_gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        calls.append(payload)
        if len(calls) == 1:
            raise GatewayClientSmokeError("endpoint_unreachable")
        return 200, _tool_call_response("artifact_write", {"relative_path": "provider/retry.txt", "content": "ok"})

    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", flaky_gateway_post_json)

    result = AgentLoop(
        _session(tmp_path, "https://gateway.test/v1"),
        ProviderToolChoiceModel(api_key="test-token", retry_backoff_s=0),
    ).run()

    assert result.status == "succeeded"
    assert len(calls) == 2
    assert (tmp_path / "artifacts" / "provider" / "retry.txt").read_text(encoding="utf-8") == "ok"


def test_provider_tool_choice_model_blocks_malformed_explicit_api_protocol(tmp_path: Path) -> None:
    session = _session(tmp_path, "https://gateway.test/v1")
    malformed = replace(session, endpoint=replace(session.endpoint, api_protocol="definitely_not_a_protocol"))
    model = ProviderToolChoiceModel(api_key="test-token", retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="invalid_api_protocol=definitely_not_a_protocol"):
        model.choose_tools(malformed, malformed.model_visible_tool_schemas())


def test_provider_tool_choice_model_accepts_final_text_without_tool_call(tmp_path: Path) -> None:
    with _ToolChoiceGateway({"output_text": "No tool needed."}) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "succeeded"
    assert result.blockers == ()
    assert result.selected_tools == ()
    assert result.final_output == "No tool needed."
    assert result.session.messages[-1] == {"role": "assistant", "content": "No tool needed."}


def test_provider_tool_choice_model_accepts_sse_output_text_without_tool_call(tmp_path: Path) -> None:
    sse_body = "\n\n".join(
        (
            'data: {"type":"response.output_text.delta","delta":"No tool"}',
            'data: {"type":"response.output_text.delta","delta":" needed."}',
            'data: {"type":"response.output_text.done","text":"No tool needed."}',
            "data: [DONE]",
        )
    )
    with _ToolChoiceGateway(sse_body, response_content_type="text/event-stream") as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "succeeded"
    assert result.blockers == ()
    assert result.selected_tools == ()
    assert result.final_output == "No tool needed."


def test_provider_tool_choice_model_blocks_unknown_tool_before_execution(tmp_path: Path) -> None:
    with _ToolChoiceGateway(_tool_call_response("delete_everything", {})) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "blocked"
    assert result.blockers == ("unknown_model_tool_selected",)
    assert result.tool_results == ()


def test_provider_tool_choice_model_blocks_unsafe_or_non_executable_tool_before_execution(tmp_path: Path) -> None:
    with _ToolChoiceGateway(_tool_call_response("validate_simulation_request", {})) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "blocked"
    assert result.blockers == ("unsafe_model_tool_selected",)
    assert result.tool_results == ()


def test_provider_tool_choice_model_blocks_malformed_tool_arguments(tmp_path: Path) -> None:
    with _ToolChoiceGateway({"output": [{"type": "tool_call", "name": "artifact_write", "arguments": "not-json"}]}) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "blocked"
    assert result.blockers == ("malformed_model_tool_call",)
    assert result.tool_results == ()


def _session(
    tmp_path: Path,
    base_url: str,
    *,
    provider: str = "oauth_gateway",
    auth_mode: str = "gateway",
) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": base_url,
            "auth_mode": auth_mode,
            "api_key_env": "MODEL_GATEWAY_TOKEN",
        }
    )
    return AsaAgentSession(
        run_id="provider-run",
        session_id="provider-session",
        agent_id="orchestrator",
        user_goal="Write provider-selected evidence",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
        role_prompt=load_domain_role_prompt("orchestrator"),
    )


def _tool_call_response(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "output": [
            {
                "type": "tool_call",
                "name": name,
                "arguments": arguments,
            }
        ]
    }


def _fake_codex_token(account_id: str) -> str:
    def encode(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return ".".join(
        (
            encode({"alg": "none"}),
            encode({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}),
            "",
        )
    )


class _ToolChoiceGateway:
    def __init__(
        self,
        response_payload: dict[str, object] | str,
        *,
        expected_path: str = "/v1/responses",
        expected_token: str = "test-token",
        response_content_type: str = "application/json",
    ) -> None:
        self._handler = type(
            "_ToolChoiceGatewayHandler",
            (_ToolChoiceGatewayHandler,),
            {
                "response_payload": response_payload,
                "response_content_type": response_content_type,
                "request_body": None,
                "request_headers": {},
                "expected_path": expected_path,
                "expected_token": expected_token,
            },
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def backend_base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def request_body(self) -> dict[str, object]:
        body = self._handler.request_body
        assert isinstance(body, dict)
        return body

    @property
    def request_headers(self) -> dict[str, str]:
        return dict(self._handler.request_headers)

    def __enter__(self) -> "_ToolChoiceGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _ToolChoiceGatewayHandler(BaseHTTPRequestHandler):
    response_payload: dict[str, object] | str = {}
    response_content_type = "application/json"
    request_body: dict[str, object] | None = None
    request_headers: dict[str, str] = {}
    expected_path = "/v1/responses"
    expected_token = "test-token"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != self.expected_path:
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        if self.headers.get("authorization") != f"Bearer {self.expected_token}":
            self._write({"error": {"code": "missing_gateway_credentials"}}, status=401)
            return
        length = int(self.headers["content-length"])
        self.__class__.request_headers = {
            "openai_beta": self.headers.get("OpenAI-Beta", ""),
            "originator": self.headers.get("originator", ""),
            "account_id": self.headers.get("chatgpt-account-id", ""),
        }
        self.__class__.request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        self._write(self.response_payload)

    def _write(self, payload: dict[str, object] | str, status: int = 200) -> None:
        if isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", self.response_content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
