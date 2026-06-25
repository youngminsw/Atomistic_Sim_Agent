from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sim_agent.agent_runtime import (
    CompactionRequest,
    append_agent_message,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.live_agent_context import live_turn_handle_with_model_override
from sim_agent.agent_runtime.live_agent_turn import _agent_loop_session, run_live_agent_turn
from sim_agent.agents_sdk_runtime import assemble_provider_context
from sim_agent.cli.tui_state import initial_state
from sim_agent.runtime_config import (
    AgentModelRuntimeConfig,
    RUNTIME_CONFIG_ENV,
    default_runtime_config,
    save_runtime_config,
)


def test_live_agent_turn_uses_agent_override_and_hydrates_prompt_layers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(RUNTIME_CONFIG_ENV, str(tmp_path / "runtime-config.json"))
    config = default_runtime_config()
    override = AgentModelRuntimeConfig(
        agent_id="md_agent",
        provider="static",
        model="g002-md-override",
        reasoning_effort="xhigh",
        base_url="https://model-gateway.local/v1",
        auth_mode="none",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )
    save_runtime_config(replace(config, agent_model_overrides=(override,)))
    state = initial_state(tmp_path / "session")

    first = run_live_agent_turn(state.session_dir, "md_agent", "first MD turn")
    second = run_live_agent_turn(state.session_dir, "md_agent", "second MD turn")

    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    session = _agent_loop_session(live_turn_handle_with_model_override(handle), "third MD turn")
    context = assemble_provider_context(session)

    assert first.agent_session_id == second.agent_session_id
    assert second.model_id == "static/g002-md-override"
    assert session.endpoint.model == "g002-md-override"
    assert session.endpoint.reasoning_effort == "xhigh"
    assert "domain_role" in context.layer_kinds()
    assert "project_guidance" in context.layer_kinds()
    assert "/md" in context.instructions
    assert "Plan and verify MD work" in context.instructions
    assert "agent_loop_completed" in context.instructions
    assert {"role": "user", "content": "first MD turn"} in context.openai_responses_input()
    assert {"role": "user", "content": "second MD turn"} in context.openai_responses_input()


def test_live_agent_session_ignores_unreplayed_manual_compaction_until_replayed(
    tmp_path: Path,
) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "qa_agent", "user", "seed QA turn")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(agent_id="qa_agent", compact_id="manual-qa-edge", summary="validated QA summary"),
    )

    unreplayed_handle = load_agent_registry(state.session_dir).handles["qa_agent"]
    unreplayed_session = _agent_loop_session(unreplayed_handle, "next QA turn")

    replayed = replay_agent_compaction(state.session_dir, "qa_agent")
    replayed_handle = load_agent_registry(state.session_dir).handles["qa_agent"]
    replayed_session = _agent_loop_session(replayed_handle, "next QA turn")

    assert unreplayed_session.compact_summary == ""
    assert unreplayed_session.messages == []
    assert unreplayed_session.provider_context_blocker == "manual_replay_required"
    assert replayed.status == "succeeded"
    assert "validated QA summary" in replayed_session.compact_summary
    assert "Another language model started to solve this problem" in replayed_session.compact_summary
