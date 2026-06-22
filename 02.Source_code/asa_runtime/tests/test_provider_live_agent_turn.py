from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from sim_agent.agent_runtime import (
    CompactionRequest,
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_agent_message,
    compact_agent_session,
    open_global_session,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.live_agent_turn import run_live_agent_turn


def test_live_agent_turn_uses_provider_model_for_connected_session(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        record = open_global_session(
            GlobalSessionOpenRequest(
                requested_dir=tmp_path / "session",
                default_root=tmp_path,
                model=_model(gateway.base_url, auth_mode="gateway"),
            )
        ).record

        result = run_live_agent_turn(record.session_dir, "orchestrator", "Provider live turn")

    body = gateway.request_body
    assert result.status == "succeeded"
    assert result.model_id == "oauth_gateway/gpt-5.5"
    assert result.selected_tools == ("artifact_write",)
    assert body["model"] == "gpt-5.5"
    assert body["tools"]
    evidence = record.paths.agent_sessions / "orchestrator" / "artifacts" / "live_agent_turn" / "provider.txt"
    assert evidence.read_text(encoding="utf-8") == "provider-live-ok"


def test_live_agent_turn_sends_persistent_agent_transcript_to_provider(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        record = open_global_session(
            GlobalSessionOpenRequest(
                requested_dir=tmp_path / "session",
                default_root=tmp_path,
                model=_model(gateway.base_url, auth_mode="gateway"),
            )
        ).record

        first = run_live_agent_turn(record.session_dir, "orchestrator", "First provider turn")
        second = run_live_agent_turn(record.session_dir, "orchestrator", "Second provider turn")

    body = gateway.request_body
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert body["input"][:3] == [
        {"role": "user", "content": "First provider turn"},
        {"role": "assistant", "content": "agent loop completed with status succeeded"},
        {"role": "user", "content": "Second provider turn"},
    ]
    assert "Coordinate the ASA runtime" in body["instructions"]
    messages_path = record.paths.agent_sessions / "orchestrator" / "messages.jsonl"
    assert "First provider turn" in messages_path.read_text(encoding="utf-8")
    assert "Second provider turn" in messages_path.read_text(encoding="utf-8")


def test_live_agent_turn_injects_only_validated_compact_summary(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        record = open_global_session(
            GlobalSessionOpenRequest(
                requested_dir=tmp_path / "session",
                default_root=tmp_path,
                model=_model(gateway.base_url, auth_mode="gateway"),
            )
        ).record
        append_agent_message(record.session_dir, "orchestrator", "user", "Old context before compaction")
        compact_agent_session(
            record.session_dir,
            CompactionRequest(
                agent_id="orchestrator",
                compact_id="compact-orch-001",
                summary="Validated compact summary should reach the provider.",
            ),
        )

        first = run_live_agent_turn(record.session_dir, "orchestrator", "Before replay")
        body_before_replay = dict(gateway.request_body)
        replayed = replay_agent_compaction(record.session_dir, "orchestrator")
        second = run_live_agent_turn(record.session_dir, "orchestrator", "After replay")
        body_after_replay = gateway.request_body

    assert first.status == "succeeded"
    assert replayed.status == "succeeded"
    assert second.status == "succeeded"
    assert "Validated compact summary should reach the provider." not in body_before_replay["instructions"]
    assert "Validated compact summary should reach the provider." in body_after_replay["instructions"]
    assert body_after_replay["input"][-1] == {"role": "user", "content": "After replay"}


def test_live_agent_turn_keeps_static_model_for_offline_auth_none_session(tmp_path: Path) -> None:
    record = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=tmp_path / "session",
            default_root=tmp_path,
            model=_model("http://127.0.0.1:9/v1", auth_mode="none", provider="local_gateway", name="offline-static"),
        )
    ).record

    result = run_live_agent_turn(record.session_dir, "orchestrator", "Offline static turn")

    assert result.status == "succeeded"
    assert result.model_id == "local_gateway/offline-static"
    assert result.selected_tools == ("artifact_write",)
    evidence = record.paths.agent_sessions / "orchestrator" / "artifacts" / "live_agent_turn" / "evidence.txt"
    assert "Offline static turn" in evidence.read_text(encoding="utf-8")


def _model(
    base_url: str,
    *,
    auth_mode: str,
    provider: str = "oauth_gateway",
    name: str = "gpt-5.5",
) -> GlobalSessionModel:
    return GlobalSessionModel(
        provider=provider,
        name=name,
        reasoning_effort="high",
        base_url=base_url,
        auth_mode=auth_mode,
        api_key_env="MODEL_GATEWAY_TOKEN",
    )


class _LiveTurnGateway:
    def __init__(self) -> None:
        self._handler = type("_LiveTurnGatewayHandler", (_LiveTurnGatewayHandler,), {"request_body": None})
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

    def __enter__(self) -> "_LiveTurnGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _LiveTurnGatewayHandler(BaseHTTPRequestHandler):
    request_body: dict[str, object] | None = None

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self._write({"error": {"code": "not_found"}}, status=404)
            return
        length = int(self.headers["content-length"])
        self.__class__.request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        self._write(
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "artifact_write",
                        "arguments": {
                            "relative_path": "live_agent_turn/provider.txt",
                            "content": "provider-live-ok",
                        },
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
