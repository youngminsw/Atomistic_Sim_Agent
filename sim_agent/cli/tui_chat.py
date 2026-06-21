from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str
from sim_agent.schemas.errors import SchemaValidationError

from .tui_chat_render import write_chat_deck
from .tui_direct_agent import DirectAgentChatRequest, run_direct_agent_chat
from .tui_paths import display_path
from .tui_semantic import write_semantic_line, write_semantic_lines
from .tui_state import TuiState, append_event


ChatRole = Literal["user", "assistant", "system"]
CHAT_TRANSCRIPT_NAME: Final = "asa_chat_transcript.jsonl"
CHAT_RUNS_DIR_NAME: Final = "chat-runs"
AGENT_TARGETS: Final = ("orchestrator", *(role.role_id for role in AGENT_ROLES))


@dataclass(frozen=True, slots=True)
class ChatMessage:
    at: float
    role: ChatRole
    content: str
    target: str | None


@dataclass(frozen=True, slots=True)
class ChatTranscriptView:
    messages: tuple[ChatMessage, ...]
    message_count: int
    corrupt_lines: int
    path: Path


@dataclass(frozen=True, slots=True)
class ChatHudSummary:
    message_count: int
    corrupt_lines: int
    last_role: str
    last_target: str
    path: Path


@dataclass(frozen=True, slots=True)
class TargetedChatInput:
    target: str
    message: str
    run_tokens: tuple[str, ...]
    explicit: bool


RunHandler = Callable[[Sequence[str], TuiState, TextIO], TuiState]


def handle_chat(args: Sequence[str], state: TuiState, output_stream: TextIO, run_handler: RunHandler) -> TuiState:
    if not args:
        write_chat_window(state, output_stream)
        return state
    match args[0]:
        case "clear":
            _transcript_path(state).unlink(missing_ok=True)
            append_event(state, "chat_cleared", "Chat transcript cleared")
            write_semantic_line(output_stream, "chat_cleared=true")
            write_chat_window(state, output_stream)
            return state
        case _:
            return handle_chat_message(args, state, output_stream, run_handler)


def handle_chat_message(
    args: Sequence[str],
    state: TuiState,
    output_stream: TextIO,
    run_handler: RunHandler,
) -> TuiState:
    chat_input = _targeted_input(args)
    if not chat_input.message:
        write_semantic_line(output_stream, "chat_error=missing_message")
        write_chat_window(state, output_stream)
        return state
    append_chat_message(state, "user", chat_input.message, chat_input.target)
    if chat_input.explicit:
        return _handle_direct_agent_message(chat_input, state, output_stream)
    next_state = run_handler(_run_args(chat_input, state), state, output_stream)
    summary = "Orchestrator could not prepare a run; inspect the preceding runtime lines."
    if next_state.last_run_ledger != state.last_run_ledger and next_state.last_run_ledger is not None:
        summary = f"Orchestrator prepared run ledger {next_state.last_run_ledger.name}."
    append_chat_message(next_state, "assistant", summary, "orchestrator")
    write_chat_window(next_state, output_stream)
    return next_state


def append_chat_message(state: TuiState, role: ChatRole, content: str, target: str | None) -> Path:
    state.session_dir.mkdir(parents=True, exist_ok=True)
    path = _transcript_path(state)
    payload: JsonMap = {
        "at": time.time(),
        "role": role,
        "content": _clean_text(content),
        "target": target or "",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def read_chat_transcript(state: TuiState, *, limit: int = 8) -> ChatTranscriptView:
    path = _transcript_path(state)
    if not path.is_file():
        return ChatTranscriptView((), 0, 0, path)
    messages: list[ChatMessage] = []
    corrupt_lines = 0
    for line in path.read_bytes().splitlines():
        try:
            messages.append(_chat_message_from_json(line))
        except (UnicodeDecodeError, json.JSONDecodeError, SchemaValidationError, TypeError):
            corrupt_lines += 1
    visible = tuple(messages[-limit:]) if limit > 0 else tuple(messages)
    return ChatTranscriptView(visible, len(messages), corrupt_lines, path)


def chat_hud_summary(state: TuiState) -> ChatHudSummary:
    view = read_chat_transcript(state, limit=1)
    if not view.messages:
        return ChatHudSummary(0, view.corrupt_lines, "-", "-", view.path)
    latest = view.messages[-1]
    return ChatHudSummary(view.message_count, view.corrupt_lines, latest.role, latest.target or "-", view.path)


def write_chat_window(state: TuiState, output_stream: TextIO) -> None:
    write_chat_deck(read_chat_transcript(state), output_stream)


def _chat_message_from_json(line: bytes) -> ChatMessage:
    payload = as_mapping(json.loads(line.decode("utf-8")), "chat_message")
    at_value = payload.get("at")
    if not isinstance(at_value, int | float) or isinstance(at_value, bool):
        raise SchemaValidationError("chat_message.at must be a number")
    target_value = payload.get("target")
    target = target_value if isinstance(target_value, str) and target_value else None
    return ChatMessage(
        at=float(at_value),
        role=_chat_role(as_str(payload.get("role"), "chat_message.role")),
        content=as_str(payload.get("content"), "chat_message.content"),
        target=target,
    )


def _chat_role(value: str) -> ChatRole:
    match value:
        case "user":
            return "user"
        case "assistant":
            return "assistant"
        case "system":
            return "system"
        case _:
            raise SchemaValidationError("chat_message.role must be user, assistant, or system")


def _targeted_input(args: Sequence[str]) -> TargetedChatInput:
    if not args:
        return TargetedChatInput("orchestrator", "", (), False)
    first = args[0]
    if first.startswith("@") and first.removeprefix("@") in AGENT_TARGETS:
        message = _clean_text(" ".join(args[1:]))
        target = first.removeprefix("@")
        return TargetedChatInput(target, message, (f"@{target} {message}",), True)
    return TargetedChatInput("orchestrator", _clean_text(" ".join(args)), tuple(args), False)


def _handle_direct_agent_message(
    chat_input: TargetedChatInput,
    state: TuiState,
    output_stream: TextIO,
) -> TuiState:
    result = run_direct_agent_chat(
        DirectAgentChatRequest(
            target=chat_input.target,
            message=chat_input.message,
            session_id=state.session_id,
            session_dir=state.session_dir,
        )
    )
    append_event(state, "agent_direct_route", chat_input.target)
    write_semantic_lines(
        output_stream,
        (
            f"agent_direct_route={chat_input.target}",
            f"agent_session_id={result.agent_session_id}",
            f"agent_session_path={display_path(result.agent_session_path)}",
            f"agent_loop_status={result.turn_status}",
            f"agent_loop_model={result.model_id}",
            f"agent_loop_tools={','.join(result.selected_tools)}",
        ),
    )
    append_chat_message(
        state,
        "assistant",
        result.assistant_content,
        chat_input.target,
    )
    write_chat_window(state, output_stream)
    return state


def _run_args(chat_input: TargetedChatInput, state: TuiState) -> tuple[str, ...]:
    if "--output-dir" in chat_input.run_tokens:
        return chat_input.run_tokens
    return (*chat_input.run_tokens, "--output-dir", str(state.session_dir / CHAT_RUNS_DIR_NAME))


def _transcript_path(state: TuiState) -> Path:
    return state.session_dir / CHAT_TRANSCRIPT_NAME


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())[:2000]
