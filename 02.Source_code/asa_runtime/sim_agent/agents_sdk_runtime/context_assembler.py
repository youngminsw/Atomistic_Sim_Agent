from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Literal

from sim_agent.agent_runtime.compaction_policy import ProviderContextCompactionBlocked, int_value
from sim_agent.schemas._parse import JsonMap

from .agent_loop import AsaAgentSession
from .markdown_skills import skill_context_body
from .prompt_assets import load_common_system_prompt, load_workflow_policy_prompt
from .prompt_layers import PromptLayer, PromptLayerKind, prompt_layer


ProviderRole = Literal["user", "assistant"]
JsonTextValue = str | int | float | bool | None | JsonMap | list[JsonMap]
MAX_CONTEXT_MESSAGES: Final = 24
MAX_TOOL_HISTORY: Final = 8


@dataclass(frozen=True, slots=True)
class ProviderPromptContext:
    layers: tuple[PromptLayer, ...]
    messages: tuple[JsonMap, ...]

    @property
    def instructions(self) -> str:
        return "\n\n".join(layer.instruction_section() for layer in self.layers)

    def layer_kinds(self) -> tuple[str, ...]:
        return tuple(layer.kind for layer in self.layers)

    def layers_json(self) -> list[JsonMap]:
        return [layer.to_json() for layer in self.layers]

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
    if session.provider_context_blocker:
        raise ProviderContextCompactionBlocked(session.provider_context_blocker)
    messages = _conversation_messages(session)
    return ProviderPromptContext(
        layers=_prompt_layers(session),
        messages=messages,
    )


def _prompt_layers(session: AsaAgentSession) -> tuple[PromptLayer, ...]:
    layers: list[PromptLayer] = [
        PromptLayer(
            kind="system_policy",
            title="ASA common system policy",
            content=load_common_system_prompt(),
            source="asa.prompts.system.common_system",
        ),
        PromptLayer(
            kind="workflow_policy",
            title="Workflow policy",
            content=session.workflow_policy.strip() or load_workflow_policy_prompt(),
            source="asa.runtime.workflow" if session.workflow_policy.strip() else "asa.prompts.system.workflow_policy",
        ),
    ]
    if session.role_prompt:
        _append_layer(
            layers,
            _role_prompt_kind(session),
            _role_prompt_title(session),
            session.role_prompt,
            f"agent:{session.agent_id}",
        )
    if session.project_guidance:
        _append_layer(layers, "project_guidance", "Project guidance", session.project_guidance, "asa.project")
    if session.compact_summary:
        _append_layer(layers, "compact_summary", "Compact summary", session.compact_summary, "asa.compaction")
    if session.caller_context:
        _append_layer(layers, "caller_context", "Bounded caller context", session.caller_context, "asa.subagent.caller_context")
    skill_contexts = _skill_contexts(session)
    if session.skills or skill_contexts:
        _append_layer(layers, "skills", "Active skill/workflow surfaces", _skills_text(session.skills, skill_contexts), "asa.skills")
    if session.workflow_state:
        _append_layer(layers, "workflow_state", "Workflow state", _json_text(session.workflow_state), "asa.workflow_state")
    if session.ledger_facts:
        _append_layer(layers, "ledger_facts", "Evidence ledger facts", _json_text(session.ledger_facts[-MAX_TOOL_HISTORY:]), "asa.ledger")
    if session.tool_history:
        _append_layer(layers, "tool_history", "Recent tool history", _tool_history_text(session.tool_history[-MAX_TOOL_HISTORY:]), "asa.tools")
    return tuple(layers)


def _append_layer(
    layers: list[PromptLayer],
    kind: PromptLayerKind,
    title: str,
    content: str,
    source: str,
) -> None:
    layer = prompt_layer(kind, title, content, source)
    if layer is not None:
        layers.append(layer)


def _role_prompt_kind(session: AsaAgentSession) -> PromptLayerKind:
    if session.role_prompt_kind == "subagent_role":
        return "subagent_role"
    return "domain_role"


def _role_prompt_title(session: AsaAgentSession) -> str:
    if _role_prompt_kind(session) == "subagent_role":
        return "Bounded subagent role"
    return "Domain agent role"


def _skills_text(skills: tuple[str, ...], skill_contexts: tuple[str, ...]) -> str:
    names = tuple(skill for skill in skills if isinstance(skill, str) and skill)
    named = tuple(f"- {skill}" for skill in names)
    return "\n\n".join(("\n".join(named), *skill_contexts)).strip()


def _skill_contexts(session: AsaAgentSession) -> tuple[str, ...]:
    contexts: list[str] = []
    for record in _provider_visible_records(session):
        context = skill_context_body(record.get("content")) if record.get("role") == "system" else ""
        if context:
            contexts.append(context)
    return tuple(contexts)


def _conversation_messages(session: AsaAgentSession) -> tuple[JsonMap, ...]:
    parsed_messages: list[JsonMap] = []
    for record in _provider_visible_records(session):
        message = _message_from_record(record)
        if message is not None:
            parsed_messages.append(message)
    messages = tuple(parsed_messages)
    if not messages or _content_text(messages[-1].get("content")) != session.user_goal:
        messages = (*messages, {"role": "user", "content": session.user_goal})
    if int_value(session.compaction_metadata, "first_kept_message_sequence") > 0:
        return messages
    return messages[-MAX_CONTEXT_MESSAGES:]


def _provider_visible_records(session: AsaAgentSession) -> tuple[JsonMap, ...]:
    first_kept = int_value(session.compaction_metadata, "first_kept_message_sequence")
    if first_kept <= 0:
        return tuple(session.messages)
    visible: list[JsonMap] = []
    for record in session.messages:
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ProviderContextCompactionBlocked("missing_compaction_message_sequence")
        if sequence >= first_kept:
            visible.append(record)
    return tuple(visible)


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


def _content_text(value: JsonTextValue) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _json_text(value)


def _json_text(value: JsonTextValue | list[JsonTextValue]) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return ""
