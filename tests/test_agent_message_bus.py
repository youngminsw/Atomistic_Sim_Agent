from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agent_runtime import (
    ReplyAgentMessageRequest,
    SendAgentMessageRequest,
    ack_agent_message,
    read_agent_message,
    reply_agent_message,
    send_agent_message,
)
from sim_agent.cli.tui_state import initial_state


def test_message_bus_send_ack_read_reply_records_are_append_only(tmp_path: Path) -> None:
    state = initial_state(tmp_path)

    sent = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="md_agent",
            content="check MD event coverage",
            thread_id="thread-md",
            message_id="msg-001",
        ),
    )
    acked = ack_agent_message(state.session_dir, message_id="msg-001", by_agent="md_agent")
    read = read_agent_message(state.session_dir, message_id="msg-001", by_agent="md_agent")
    replied = reply_agent_message(
        state.session_dir,
        ReplyAgentMessageRequest(message_id="msg-001", by_agent="md_agent", content="coverage checked"),
    )

    records = _jsonl(state.session_dir / "message_bus" / "messages.jsonl")
    assert sent.status == "succeeded"
    assert acked.status == "succeeded"
    assert read.status == "succeeded"
    assert replied.status == "succeeded"
    assert [record["record_type"] for record in records] == ["send", "ack", "read", "reply"]
    assert [record["status"] for record in records] == ["sent", "acknowledged", "read", "replied"]
    assert records[0]["from"] == "orchestrator"
    assert records[0]["to"] == "md_agent"
    assert records[0]["thread_id"] == "thread-md"


def test_message_bus_blocks_unknown_duplicate_blocked_stale_and_missing_bus(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="first",
            thread_id="thread-qa",
            message_id="dup-msg",
        ),
    )
    duplicate = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="duplicate",
            thread_id="thread-qa",
            message_id="dup-msg",
        ),
    )
    unknown = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="unknown_agent",
            content="unknown",
            thread_id="thread-bad",
            message_id="unknown-msg",
        ),
    )
    blocked = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="blocked",
            thread_id="thread-qa",
            message_id="blocked-msg",
            blocked_targets=("qa_agent",),
        ),
    )
    (state.session_dir / "message_bus" / "agent_message.lock").write_text("stale", encoding="utf-8")
    stale = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="stale",
            thread_id="thread-qa",
            message_id="stale-msg",
        ),
    )
    (state.session_dir / "message_bus" / "agent_message.lock").unlink()
    for child in (state.session_dir / "message_bus").iterdir():
        child.unlink()
    (state.session_dir / "message_bus").rmdir()
    missing = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="missing",
            thread_id="thread-qa",
            message_id="missing-msg",
        ),
    )

    assert duplicate.blocker == "duplicate_message_id"
    assert unknown.blocker == "unknown_agent"
    assert blocked.blocker == "blocked_target"
    assert stale.blocker == "stale_lock"
    assert missing.blocker == "message_bus_missing"
    error_records = _jsonl(state.session_dir / "message_bus_errors.jsonl")
    assert [record["blocker"] for record in error_records] == [
        "duplicate_message_id",
        "unknown_agent",
        "blocked_target",
        "stale_lock",
        "message_bus_missing",
    ]


def test_message_bus_blocks_state_transition_from_wrong_recipient(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="md_agent",
            content="md only",
            thread_id="thread-md",
            message_id="msg-md-only",
        ),
    )

    acked = ack_agent_message(state.session_dir, message_id="msg-md-only", by_agent="qa_agent")
    read = read_agent_message(state.session_dir, message_id="msg-md-only", by_agent="qa_agent")
    replied = reply_agent_message(
        state.session_dir,
        ReplyAgentMessageRequest(message_id="msg-md-only", by_agent="qa_agent", content="wrong target"),
    )

    records = _jsonl(state.session_dir / "message_bus" / "messages.jsonl")
    errors = _jsonl(state.session_dir / "message_bus_errors.jsonl")
    assert acked.blocker == "wrong_recipient"
    assert read.blocker == "wrong_recipient"
    assert replied.blocker == "wrong_recipient"
    assert [record["record_type"] for record in records] == ["send"]
    assert [record["blocker"] for record in errors[-3:]] == ["wrong_recipient", "wrong_recipient", "wrong_recipient"]


def test_message_bus_blocks_corrupt_bus_ledger_with_durable_error(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    (state.session_dir / "message_bus" / "messages.jsonl").write_text("{broken-json\n", encoding="utf-8")

    sent = send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="qa_agent",
            content="after corruption",
            thread_id="thread-qa",
            message_id="msg-after-corruption",
        ),
    )
    acked = ack_agent_message(state.session_dir, message_id="msg-after-corruption", by_agent="qa_agent")

    errors = _jsonl(state.session_dir / "message_bus_errors.jsonl")
    assert sent.blocker == "corrupt_message_bus"
    assert acked.blocker == "corrupt_message_bus"
    assert [record["blocker"] for record in errors[-2:]] == ["corrupt_message_bus", "corrupt_message_bus"]


def test_message_bus_blocks_state_transition_when_tail_is_corrupt(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    send_agent_message(
        state.session_dir,
        SendAgentMessageRequest(
            from_agent="orchestrator",
            to_agent="md_agent",
            content="valid first record",
            thread_id="thread-md",
            message_id="msg-before-corrupt-tail",
        ),
    )
    with (state.session_dir / "message_bus" / "messages.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{broken-tail\n")

    acked = ack_agent_message(state.session_dir, message_id="msg-before-corrupt-tail", by_agent="md_agent")

    records = _jsonl_allow_corrupt_tail(state.session_dir / "message_bus" / "messages.jsonl")
    errors = _jsonl(state.session_dir / "message_bus_errors.jsonl")
    assert acked.blocker == "corrupt_message_bus"
    assert [record["record_type"] for record in records] == ["send"]
    assert errors[-1]["blocker"] == "corrupt_message_bus"


def test_agent_message_runtime_tool_writes_bus_and_tool_ledger(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    registry = default_tool_registry()

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="agent_message",
            arguments={
                "action": "send",
                "from_agent": "orchestrator",
                "to_agent": "md_agent",
                "content": "tool mediated message",
                "thread_id": "tool-thread",
                "message_id": "tool-msg-001",
            },
            run_id="tool-msg-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert result.status == "succeeded"
    assert result.output["message_id"] == "tool-msg-001"
    assert result.output["status"] == "sent"
    assert "agent_message" in registry.tool_names
    assert (state.session_dir / result.artifact_ref).is_file()
    assert _jsonl(state.session_dir / "message_bus" / "messages.jsonl")[-1]["message_id"] == "tool-msg-001"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _jsonl_allow_corrupt_tail(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records
