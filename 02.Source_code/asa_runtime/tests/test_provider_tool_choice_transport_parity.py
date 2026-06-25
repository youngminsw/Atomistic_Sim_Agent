from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AgentLoop, AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.llm_endpoints import ModelProviderConfig


@pytest.mark.parametrize(
    ("provider", "expected_path", "response", "artifact", "checks"),
    (
        pytest.param(
            "deepseek",
            "/v1/chat/completions",
            {"choices": [{"message": {"tool_calls": [{"function": {"name": "artifact_write", "arguments": json.dumps({"relative_path": "provider/chat.txt", "content": "ok"})}}]}}]},
            "provider/chat.txt",
            (("messages.0.role", "system"), ("tool:artifact_write.type", "function"), ("tool:artifact_write.function.name", "artifact_write")),
            id="openai-compatible-chat",
        ),
        pytest.param(
            "anthropic",
            "/v1/messages",
            {"content": [{"type": "tool_use", "name": "artifact_write", "input": {"relative_path": "provider/anthropic.txt", "content": "ok"}}]},
            "provider/anthropic.txt",
            (("tool_choice.type", "auto"), ("tools.0.input_schema.type", "object"), ("messages.0.role", "user")),
            id="anthropic-messages",
        ),
        pytest.param(
            "google-gemini-cli",
            "/v1beta/models/gpt-5.5:generateContent",
            {"candidates": [{"content": {"parts": [{"functionCall": {"name": "artifact_write", "args": {"relative_path": "provider/gemini.txt", "content": "ok"}}}]}}]},
            "provider/gemini.txt",
            (("toolConfig.functionCallingConfig.mode", "AUTO"), ("gemini_tool:artifact_write.name", "artifact_write"), ("contents.0.role", "user")),
            id="gemini-generate-content",
        ),
    ),
)
def test_provider_tool_choice_model_posts_protocol_specific_payloads(
    tmp_path: Path,
    provider: str,
    expected_path: str,
    response: dict[str, object],
    artifact: str,
    checks: tuple[tuple[str, object], ...],
) -> None:
    with _ToolChoiceGateway(response, expected_path=expected_path) as gateway:
        session = _session(tmp_path, gateway.base_url, provider)
        result = AgentLoop(session, ProviderToolChoiceModel(api_key="test-token")).run()

    assert result.status == "succeeded"
    for path, expected in checks:
        assert _nested_value(gateway.request_body, path) == expected
    assert (tmp_path / "artifacts" / artifact).read_text(encoding="utf-8") == "ok"


def _session(tmp_path: Path, base_url: str, provider: str) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": base_url,
            "auth_mode": "gateway",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
        }
    )
    return AsaAgentSession(
        run_id="provider-parity-run",
        session_id="provider-parity-session",
        agent_id="orchestrator",
        user_goal="Write provider-selected evidence",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )


def _nested_value(payload: dict[str, object], path: str) -> object:
    if path.startswith("tool:"):
        tool_name, rest = path.removeprefix("tool:").split(".", 1)
        return _nested_value(_openai_tool(payload, tool_name), rest)
    if path.startswith("gemini_tool:"):
        tool_name, rest = path.removeprefix("gemini_tool:").split(".", 1)
        return _nested_value(_gemini_tool(payload, tool_name), rest)
    current: object = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
            continue
        assert isinstance(current, list)
        current = current[int(part)]
    return current


def _openai_tool(payload: dict[str, object], name: str) -> dict[str, object]:
    tools = payload.get("tools")
    assert isinstance(tools, list)
    for tool in tools:
        assert isinstance(tool, dict)
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return tool
    raise AssertionError(f"missing tool schema: {name}")


def _gemini_tool(payload: dict[str, object], name: str) -> dict[str, object]:
    tools = payload.get("tools")
    assert isinstance(tools, list)
    for tool_group in tools:
        assert isinstance(tool_group, dict)
        declarations = tool_group.get("functionDeclarations")
        assert isinstance(declarations, list)
        for declaration in declarations:
            assert isinstance(declaration, dict)
            if declaration.get("name") == name:
                return declaration
    raise AssertionError(f"missing gemini tool schema: {name}")


class _ToolChoiceGateway:
    def __init__(self, response_payload: dict[str, object], *, expected_path: str) -> None:
        self._handler = type(
            "_ToolChoiceGatewayHandler",
            (_ToolChoiceGatewayHandler,),
            {"response_payload": response_payload, "request_body": None, "expected_path": expected_path},
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
    expected_path = ""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != self.expected_path:
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
