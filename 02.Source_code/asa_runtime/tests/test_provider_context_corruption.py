from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from sim_agent.agent_runtime import (
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    GlobalSessionRecord,
    append_agent_message,
    load_agent_registry,
    open_global_session,
)
from sim_agent.agent_runtime.compaction_policy import ProviderContextCompactionBlocked
from sim_agent.agent_runtime.live_agent_turn import run_live_agent_turn
from sim_agent.agent_runtime.provider_context_projection import provider_visible_agent_context


def test_corrupt_uncompacted_messages_blocks_provider_context(tmp_path: Path) -> None:
    # Given: an uncompacted agent session whose messages ledger contains invalid JSONL.
    record = _open_record(tmp_path, "http://127.0.0.1:9/v1")
    agent_dir = record.paths.agent_sessions / "orchestrator"
    (agent_dir / "messages.jsonl").write_text("{broken-json\n", encoding="utf-8")
    handle = load_agent_registry(record.session_dir).handles["orchestrator"]

    # When / Then: provider projection fails closed instead of returning an empty context.
    with pytest.raises(ProviderContextCompactionBlocked, match="corrupt_messages_jsonl"):
        provider_visible_agent_context(handle)


def test_valid_uncompacted_messages_project_for_provider(tmp_path: Path) -> None:
    # Given: a valid uncompacted transcript.
    record = _open_record(tmp_path, "http://127.0.0.1:9/v1")
    append_agent_message(record.session_dir, "orchestrator", "user", "valid provider context")
    handle = load_agent_registry(record.session_dir).handles["orchestrator"]

    # When: provider projection reads the uncompacted ledger.
    context = provider_visible_agent_context(handle)

    # Then: the valid message remains visible to the provider.
    assert context.messages == ({"role": "user", "content": "valid provider context", "sequence": 1},)
    assert context.compact_summary == ""
    assert context.compaction is None
    assert context.raw_message_count == 1


def test_corrupt_messages_prevents_provider_request_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a connected provider session with a corrupt uncompacted messages ledger.
    with _RecordingGateway() as gateway:
        record = _open_record(tmp_path, gateway.base_url)
        agent_dir = record.paths.agent_sessions / "orchestrator"
        (agent_dir / "messages.jsonl").write_text("{broken-json\n", encoding="utf-8")
        monkeypatch.setattr(
            "sim_agent.agent_runtime.live_agent_turn.provider_boundary_compaction_blocker",
            lambda _session_dir, _handle: "",
        )

        # When: a live provider turn tries to resume the corrupt transcript.
        result = run_live_agent_turn(record.session_dir, "orchestrator", "latest user turn")

    # Then: the turn blocks before provider request and prompt manifest emission.
    assert result.status == "blocked"
    assert result.blockers == ("corrupt_messages_jsonl",)
    assert gateway.request_count == 0
    assert not (agent_dir / "prompt_assembly_manifest.json").exists()


def _open_record(tmp_path: Path, base_url: str) -> GlobalSessionRecord:
    return open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=tmp_path / "session",
            default_root=tmp_path,
            model=GlobalSessionModel(
                provider="oauth_gateway",
                name="gpt-5.5",
                reasoning_effort="high",
                base_url=base_url,
                auth_mode="gateway",
                api_key_env="MODEL_GATEWAY_TOKEN",
            ),
        )
    ).record


class _RecordingGateway:
    def __init__(self) -> None:
        self._handler = type("_RecordingGatewayHandler", (_RecordingGatewayHandler,), {"request_count": 0})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def request_count(self) -> int:
        return self._handler.request_count

    def __enter__(self) -> "_RecordingGateway":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _RecordingGatewayHandler(BaseHTTPRequestHandler):
    request_count = 0

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        type(self).request_count += 1
        self.send_response(500)
        self.end_headers()
