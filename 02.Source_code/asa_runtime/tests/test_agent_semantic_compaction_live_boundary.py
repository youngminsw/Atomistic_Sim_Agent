from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from sim_agent.agent_runtime import (
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_agent_message,
    open_global_session,
)
from sim_agent.agent_runtime.live_agent_turn import run_live_agent_turn


OLD_RAW_SENTINEL = "SEMANTIC_OLD_RAW_MUST_STAY_ON_DISK_ONLY"
TAIL_SENTINEL = "SEMANTIC_RECENT_TAIL_STAYS_VISIBLE"
CURRENT_SENTINEL = "SEMANTIC_CURRENT_TURN_STAYS_VISIBLE"
SUMMARY_SENTINEL = "SEMANTIC_SUMMARY_REPLACES_OLD_CONTEXT"


def test_live_provider_boundary_auto_compacts_with_provider_semantic_summary(tmp_path: Path) -> None:
    with _SemanticLiveGateway() as gateway:
        record = open_global_session(
            GlobalSessionOpenRequest(
                requested_dir=tmp_path / "session",
                default_root=tmp_path,
                model=_live_model(gateway.base_url),
            )
        ).record
        append_agent_message(record.session_dir, "orchestrator", "user", OLD_RAW_SENTINEL)
        for index in range(31):
            content = TAIL_SENTINEL if index == 30 else f"semantic boundary filler {index}"
            append_agent_message(record.session_dir, "orchestrator", "assistant" if index % 2 else "user", content)

        result = run_live_agent_turn(record.session_dir, "orchestrator", CURRENT_SENTINEL)

    summary_path = record.paths.agent_sessions / "orchestrator" / "compact_summary.json"
    messages_path = record.paths.agent_sessions / "orchestrator" / "messages.jsonl"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    provider_body = gateway.request_bodies[-1]
    summary_bodies = tuple(body for body in gateway.request_bodies if not body.get("tools"))
    assert result.status == "succeeded"
    assert summary_bodies
    assert "Summarize conversations between users and AI coding assistants" in str(summary_bodies[0].get("instructions", ""))
    assert OLD_RAW_SENTINEL in repr(summary_bodies[0])
    assert OLD_RAW_SENTINEL in messages_path.read_text(encoding="utf-8")
    assert OLD_RAW_SENTINEL not in repr(provider_body)
    assert SUMMARY_SENTINEL in repr(provider_body)
    assert TAIL_SENTINEL in repr(provider_body)
    assert provider_body["input"][-1] == {"role": "user", "content": CURRENT_SENTINEL}
    assert payload["schema_version"] == "asa_agent_compact_summary_v4"
    assert payload["compact_mode"] == "auto"
    assert payload["summary_source"] == "llm_semantic"
    assert payload["provider_cache_invalidated"] is True


def _live_model(base_url: str) -> GlobalSessionModel:
    return GlobalSessionModel(
        provider="oauth_gateway",
        name="gpt-5.5",
        reasoning_effort="high",
        base_url=base_url,
        auth_mode="gateway",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )


class _SemanticLiveGateway:
    def __init__(self) -> None:
        self._handler = type("_SemanticLiveGatewayHandler", (_SemanticLiveGatewayHandler,), {"request_bodies": []})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def request_bodies(self) -> tuple[dict[str, object], ...]:
        return tuple(self._handler.request_bodies)

    def __enter__(self) -> "_SemanticLiveGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _SemanticLiveGatewayHandler(BaseHTTPRequestHandler):
    request_bodies: list[dict[str, object]] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.request_bodies.append(body)
        if body.get("tools"):
            self._write(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "artifact_write",
                            "arguments": {
                                "relative_path": "semantic_compaction/provider.txt",
                                "content": "semantic-provider-live-ok",
                            },
                        }
                    ]
                }
            )
            return
        summary_count = sum(1 for request in self.__class__.request_bodies if not request.get("tools"))
        text = f"## Goal\n- {SUMMARY_SENTINEL}" if summary_count == 1 else "I compacted semantic context."
        self._write({"output_text": text})

    def _write(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
