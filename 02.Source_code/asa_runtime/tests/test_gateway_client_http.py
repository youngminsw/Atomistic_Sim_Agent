from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from sim_agent.agents_sdk_runtime.gateway_client_http import gateway_post_json


def test_gateway_post_json_parses_responses_sse_tool_call() -> None:
    with _SseGateway() as gateway:
        status, payload = gateway_post_json(
            gateway.url,
            {"model": "gpt-5.5"},
            "test-token",
            5.0,
            {"OpenAI-Beta": "responses=experimental"},
        )

    assert status == 200
    assert gateway.openai_beta == "responses=experimental"
    assert payload["output"][0]["name"] == "artifact_write"
    assert json.loads(payload["output"][0]["arguments"]) == {"relative_path": "sse.txt", "content": "ok"}


class _SseGateway:
    def __init__(self) -> None:
        self._handler = type("_SseGatewayHandler", (_SseGatewayHandler,), {"openai_beta": ""})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/codex/responses"

    @property
    def openai_beta(self) -> str:
        return self._handler.openai_beta

    def __enter__(self) -> "_SseGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _SseGatewayHandler(BaseHTTPRequestHandler):
    openai_beta = ""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        self.__class__.openai_beta = self.headers.get("OpenAI-Beta", "")
        arguments = json.dumps({"relative_path": "sse.txt", "content": "ok"})
        added = {
            "type": "response.output_item.added",
            "item": {"id": "fc_test", "type": "function_call", "status": "in_progress", "name": "artifact_write", "arguments": ""},
        }
        done = {
            "type": "response.output_item.done",
            "item": {"id": "fc_test", "type": "function_call", "status": "completed", "name": "artifact_write", "arguments": arguments},
        }
        completed = {"type": "response.completed", "response": {"status": "completed", "output": []}}
        body = (
            f"data: {json.dumps(added)}\n\n"
            f"data: {json.dumps(done)}\n\n"
            f"data: {json.dumps(completed)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
