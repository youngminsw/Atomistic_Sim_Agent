from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Literal

from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession


ProviderRole = Literal["user", "assistant"]
MAX_CONTEXT_MESSAGES: Final = 24
MAX_TOOL_HISTORY: Final = 8


@dataclass(frozen=True, slots=True)
class ProviderPromptContext:
    instructions: str
    messages: tuple[JsonMap, ...]

    def openai_responses_input(self) -> list[JsonMap]:
        return [dict(message) for message in self.messages]

    def openai_chat_messages(self) -> list[JsonMap]:
        return [{"role": "system", "content": self.instructions}, *self.openai_responses_input()]

    def anthropic_messages(self) -> list[JsonMap]:
        return [dict(message) for message in self.messages]

    def gemini_contents(self) -> list[JsonMap]:
        contents: list[JsonMap] = []
        for message in self.messages:
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": _content_text(message.get("content"))}]})
        return contents


def assemble_provider_context(session: AsaAgentSession) -> ProviderPromptContext:
    messages = _conversation_messages(session)
    return ProviderPromptContext(
        instructions=_instructions(session),
        messages=messages,
    )


def _instructions(session: AsaAgentSession) -> str:
    sections = [
        (
            "You are the ASA runtime tool selector. Select safe executable tools when work requires action. "
            "Preserve evidence, respect workflow gates, and do not claim external simulation or GraphDB effects "
            "unless a tool result proves them."
        )
    ]
    if session.role_prompt:
        sections.append(f"Agent role:\n{session.role_prompt}")
    if session.compact_summary:
        sections.append(f"Compact summary:\n{session.compact_summary}")
    if session.skills:
        sections.append(f"Active skill/workflow surfaces:\n{', '.join(session.skills)}")
    if session.workflow_state:
        sections.append(f"Workflow state:\n{_json_text(session.workflow_state)}")
    if session.ledger_facts:
        sections.append(f"Evidence ledger facts:\n{_json_text(session.ledger_facts[-MAX_TOOL_HISTORY:])}")
    if session.tool_history:
        sections.append(f"Recent tool history:\n{_tool_history_text(session.tool_history[-MAX_TOOL_HISTORY:])}")
    return "\n\n".join(sections)


def _conversation_messages(session: AsaAgentSession) -> tuple[JsonMap, ...]:
    parsed_messages: list[JsonMap] = []
    for record in session.messages:
        message = _message_from_record(record)
        if message is not None:
            parsed_messages.append(message)
    messages = tuple(parsed_messages)
    if not messages or _content_text(messages[-1].get("content")) != session.user_goal:
        messages = (*messages, {"role": "user", "content": session.user_goal})
    return messages[-MAX_CONTEXT_MESSAGES:]


def _message_from_record(record: JsonMap) -> JsonMap | None:
    role = record.get("role")
    if role == "system":
        return None
    if role not in {"user", "assistant"}:
        return None
    content = _content_text(record.get("content"))
    if not content:
        return None
    return {"role": role, "content": content}


def _tool_history_text(history: list[JsonMap]) -> str:
    lines: list[str] = []
    for item in history:
        tool_name = item.get("tool_name")
        status = item.get("status")
        blocker = item.get("blocker")
        artifact_ref = item.get("artifact_ref")
        parts = [str(value) for value in (tool_name, status) if isinstance(value, str) and value]
        if isinstance(blocker, str) and blocker:
            parts.append(f"blocker={blocker}")
        if isinstance(artifact_ref, str) and artifact_ref:
            parts.append(f"artifact={artifact_ref}")
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(lines) if lines else "- none"


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _json_text(value)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)
