from __future__ import annotations

from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.runtime_config import load_runtime_config
from sim_agent.runtime_config_types import CompactionRuntimeConfig

from .agent_registry import AgentSessionHandle
from .compaction import AutoCompactionPolicy, auto_compact_agent_session
from .compaction_provider_summarizer import ProviderSemanticSummarizer


def provider_boundary_compaction_blocker(session_dir: Path, handle: AgentSessionHandle) -> str:
    if handle.model.provider in {"offline", "static"}:
        return ""
    compaction = load_runtime_config().compaction
    result = auto_compact_agent_session(
        session_dir,
        handle.agent_id,
        _policy(handle, compaction),
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


def _policy(handle: AgentSessionHandle, compaction: CompactionRuntimeConfig) -> AutoCompactionPolicy:
    return AutoCompactionPolicy(
        provider=handle.model.provider,
        model=handle.model.name,
        enabled=compaction.enabled,
        threshold_percent=compaction.threshold_percent,
        threshold_tokens=compaction.threshold_tokens,
        reserve_tokens=compaction.reserve_tokens,
        keep_recent_tokens=compaction.keep_recent_tokens,
        context_window_tokens=compaction.context_window_tokens,
    )
