from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.llm_endpoints import ModelProviderConfig


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
    assert "artifact_write" in tool_names
    assert body["reasoning"]["effort"] == "high"
    assert body["metadata"]["agent_id"] == "orchestrator"
    assert body["metadata"]["session_id"] == "provider-session"


def test_provider_tool_choice_model_parses_tool_calls_list_shape(tmp_path: Path) -> None:
    with _ToolChoiceGateway({"tool_calls": [{"name": "graphdb_dry_run", "arguments": {"database_name": "asa"}}]}) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "succeeded"
    assert tuple(call.tool_name for call in result.selected_tools) == ("graphdb_dry_run",)


def test_provider_tool_choice_model_blocks_final_text_only_as_no_tool(tmp_path: Path) -> None:
    with _ToolChoiceGateway({"output_text": "No tool needed."}) as gateway:
        result = AgentLoop(_session(tmp_path, gateway.base_url), ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "blocked"
    assert result.blockers == ("no_model_tool_selected",)
    assert result.selected_tools == ()


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


def _session(tmp_path: Path, base_url: str) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "oauth_gateway",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": base_url,
            "auth_mode": "gateway",
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


class _ToolChoiceGateway:
    def __init__(self, response_payload: dict[str, object]) -> None:
        self._handler = type(
            "_ToolChoiceGatewayHandler",
            (_ToolChoiceGatewayHandler,),
            {"response_payload": response_payload, "request_body": None},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def request_body(self) -> dict[str, object]:
        body = self._handler.request_body
        assert isinstance(body, dict)
        return body

    def __enter__(self) -> "_ToolChoiceGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _ToolChoiceGatewayHandler(BaseHTTPRequestHandler):
    response_payload: dict[str, object] = {}
    request_body: dict[str, object] | None = None

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        if self.headers.get("authorization") != "Bearer test-token":
            self._write({"error": {"code": "missing_gateway_credentials"}}, status=401)
            return
        length = int(self.headers["content-length"])
        self.__class__.request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        self._write(self.response_payload)

    def _write(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
