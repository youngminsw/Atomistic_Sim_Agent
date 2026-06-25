from __future__ import annotations

from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig

from .agent_registry import AgentSessionHandle
from .compaction import auto_compact_agent_session
from .compaction_provider_summarizer import ProviderSemanticSummarizer


def provider_boundary_compaction_blocker(session_dir: Path, handle: AgentSessionHandle) -> str:
    if handle.model.provider in {"offline", "static"}:
        return ""
    result = auto_compact_agent_session(
        session_dir,
        handle.agent_id,
        summarizer=ProviderSemanticSummarizer(_endpoint(handle)),
    )
    if result.status != "blocked":
        return ""
    return result.blocker or "provider_boundary_compaction_blocked"


def global_session_dir_for_handle(handle: AgentSessionHandle) -> Path:
    return handle.session_dir.parent.parent


def _endpoint(handle: AgentSessionHandle) -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": handle.model.provider,
            "model": handle.model.name,
            "reasoning_effort": handle.model.reasoning_effort,
            "base_url": handle.model.base_url,
            "auth_mode": handle.model.auth_mode,
            "api_key_env": handle.model.api_key_env,
        }
    )
