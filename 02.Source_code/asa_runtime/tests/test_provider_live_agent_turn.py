from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_agent_message,
    compact_agent_session,
    open_global_session,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.agent_runtime.live_agent_turn import run_live_agent_turn


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


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
        for index in range(29):
            role = "assistant" if index % 2 else "user"
            append_agent_message(record.session_dir, "orchestrator", role, f"Retained setup context {index}")
        compact_agent_session(
            record.session_dir,
            CompactionRequest(
                agent_id="orchestrator",
                compact_id="compact-orch-001",
                summary="Validated compact summary should reach the provider.",
            ),
            summarizer=RecordingSummarizer(
                SemanticSummaryResult(summary="Validated compact summary should reach the provider."),
                [],
            ),
            policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
        )

        first = run_live_agent_turn(record.session_dir, "orchestrator", "Before replay")
        replayed = replay_agent_compaction(record.session_dir, "orchestrator")
        second = run_live_agent_turn(record.session_dir, "orchestrator", "After replay")
        body_after_replay = gateway.request_body

    assert first.status == "blocked"
    assert first.blockers == ("manual_replay_required",)
    assert replayed.status == "succeeded"
    assert second.status == "succeeded"
    assert "Validated compact summary should reach the provider." in body_after_replay["instructions"]
    assert "Old context before compaction" not in repr(body_after_replay)
    assert body_after_replay["input"][-1] == {"role": "user", "content": "After replay"}


def test_domain_agent_resume_request_rewrites_compacted_context(tmp_path: Path) -> None:
    with _LiveTurnGateway() as gateway:
        record = open_global_session(
            GlobalSessionOpenRequest(
                requested_dir=tmp_path / "session",
                default_root=tmp_path,
                model=_model(gateway.base_url, auth_mode="gateway"),
            )
        ).record

        for agent_id in ("md_agent", "research_agent"):
            old_marker = f"{agent_id}_OLD_RAW_SHOULD_NOT_LEAK"
            tail_marker = f"{agent_id}_TAIL_CONTEXT_28"
            latest_marker = f"{agent_id}_LATEST_USER_TURN"
            summary = f"{agent_id} compact summary should replace older provider context."
            append_agent_message(record.session_dir, agent_id, "user", old_marker)
            for index in range(29):
                role = "assistant" if index % 2 else "user"
                append_agent_message(record.session_dir, agent_id, role, f"{agent_id}_TAIL_CONTEXT_{index}")
            compact_agent_session(
                record.session_dir,
                CompactionRequest(
                    agent_id=agent_id,
                    compact_id=f"compact-{agent_id}-001",
                    summary=summary,
                ),
                summarizer=RecordingSummarizer(SemanticSummaryResult(summary=summary), []),
                policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
            )
            replayed = replay_agent_compaction(record.session_dir, agent_id)
            resumed = open_global_session(
                GlobalSessionOpenRequest(
                    requested_dir=record.session_dir,
                    default_root=tmp_path,
                    model=_model(gateway.base_url, auth_mode="gateway"),
                    resume=str(record.session_dir),
                )
            )

            result = run_live_agent_turn(resumed.record.session_dir, agent_id, latest_marker)
            body = gateway.request_body
            messages_path = record.paths.agent_sessions / agent_id / "messages.jsonl"
            assert replayed.status == "succeeded"
            assert result.status == "succeeded"
            assert old_marker in messages_path.read_text(encoding="utf-8")
            assert old_marker not in repr(body)
            assert summary in body["instructions"]
            assert tail_marker in repr(body["input"])
            assert body["input"][-1] == {"role": "user", "content": latest_marker}


def test_live_agent_turn_keeps_static_model_only_for_explicit_static_session(tmp_path: Path) -> None:
    record = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=tmp_path / "session",
            default_root=tmp_path,
            model=_model("http://127.0.0.1:9/v1", auth_mode="none", provider="static", name="explicit-static"),
        )
    ).record

    result = run_live_agent_turn(record.session_dir, "orchestrator", "Explicit static turn")

    assert result.status == "succeeded"
    assert result.model_id == "static/explicit-static"
    assert result.selected_tools == ("artifact_write",)
    evidence = record.paths.agent_sessions / "orchestrator" / "artifacts" / "live_agent_turn" / "evidence.txt"
    assert "Explicit static turn" in evidence.read_text(encoding="utf-8")


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
