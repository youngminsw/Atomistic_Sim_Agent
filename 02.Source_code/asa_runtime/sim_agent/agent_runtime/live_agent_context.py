from __future__ import annotations

from dataclasses import replace

from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_specs
from sim_agent.project_layout import project_guidance_text
from sim_agent.runtime_config import AgentModelRuntimeConfig, agent_model_override_by_id, load_runtime_config
from sim_agent.schemas._parse import JsonMap

from .agent_registry import AgentSessionHandle
from .compaction_store import read_jsonl
from .global_session_types import GlobalSessionModel


def live_turn_workflow_policy() -> str:
    return (
        "Run ASA work as an evidence-gated agent session: keep the user turn separate from runtime policy, "
        "use model-visible tools for side effects, preserve receipts in the ledger, and stop at blockers."
    )


def live_turn_project_guidance(handle: AgentSessionHandle) -> str:
    base = (
        "Use the WSL ASA runtime spine for this session. Keep provider/model/auth choices typed and visible, "
        f"keep {handle.agent_id} inside its persistent global-session agent context, and do not rely on legacy aliases."
    )
    guidance = project_guidance_text()
    return "\n\n".join(part for part in (base, guidance) if part)


def live_turn_handle_with_model_override(handle: AgentSessionHandle) -> AgentSessionHandle:
    override = agent_model_override_by_id(load_runtime_config()).get(handle.agent_id)
    if override is None:
        return handle
    return replace(handle, model=_global_model_from_override(override))


def live_turn_skill_names(agent_id: str) -> tuple[str, ...]:
    return tuple(spec.command for spec in markdown_skill_specs() if spec.agent_id == agent_id)


def live_turn_workflow_state(handle: AgentSessionHandle) -> JsonMap:
    return {
        "agent_id": handle.agent_id,
        "agent_session_id": handle.agent_session_id,
        "boundary": handle.boundary,
        "messages_path": str(handle.messages_path),
        "events_path": str(handle.events_path),
    }


def live_turn_ledger_facts(handle: AgentSessionHandle) -> list[JsonMap]:
    records = read_jsonl(handle.events_path)
    if records is None:
        return []
    facts: list[JsonMap] = []
    for record in records[-12:]:
        event_type = record.get("event_type")
        summary = record.get("summary")
        if isinstance(event_type, str) and event_type:
            facts.append(
                {
                    "agent_id": handle.agent_id,
                    "event_type": event_type,
                    "summary": summary if isinstance(summary, str) else "",
                }
            )
    return facts


def _global_model_from_override(override: AgentModelRuntimeConfig) -> GlobalSessionModel:
    return GlobalSessionModel(
        provider=override.provider,
        name=override.model,
        reasoning_effort=override.reasoning_effort,
        base_url=override.base_url,
        auth_mode=override.auth_mode,
        api_key_env=override.api_key_env,
    )
