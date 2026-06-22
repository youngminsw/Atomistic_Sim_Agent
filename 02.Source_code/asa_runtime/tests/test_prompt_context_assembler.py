from __future__ import annotations

from pathlib import Path

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession, assemble_provider_context
from sim_agent.agents_sdk_runtime.provider_transport import ProviderApiProtocol, provider_transport_request
from sim_agent.llm_endpoints import ModelProviderConfig


def test_provider_context_assembles_role_compaction_transcript_and_tool_history(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        role_prompt="Act as the MD domain agent.",
        compact_summary="Earlier turn selected Ar on amorphous Si.",
        workflow_state={"gate": "request_validated"},
        skills=("asa-workflow", "runtime-safety"),
        ledger_facts=[{"evidence": "remote_capability_probe_passed"}],
        messages=[
            {"role": "system", "content": "internal note"},
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
        ],
        tool_history=[
            {
                "tool_name": "artifact_write",
                "status": "succeeded",
                "artifact_ref": "tool_ledgers/run/artifact_write.json",
            }
        ],
    )

    context = assemble_provider_context(session)

    assert "Act as the MD domain agent." in context.instructions
    assert "Earlier turn selected Ar on amorphous Si." in context.instructions
    assert "request_validated" in context.instructions
    assert "runtime-safety" in context.instructions
    assert "remote_capability_probe_passed" in context.instructions
    assert "artifact_write" in context.instructions
    assert context.openai_responses_input() == [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "latest request"},
    ]


def test_provider_transport_uses_assembled_context_for_openai_and_anthropic(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        role_prompt="Act as the QA gate agent.",
        compact_summary="QA compact summary.",
        messages=[{"role": "user", "content": "previous"}, {"role": "assistant", "content": "ack"}],
    )
    openai = provider_transport_request(session, _tools(session))

    assert openai.protocol is ProviderApiProtocol.OPENAI_RESPONSES
    assert "Act as the QA gate agent." in openai.payload["instructions"]
    assert "QA compact summary." in openai.payload["instructions"]
    assert openai.payload["input"][-1] == {"role": "user", "content": "latest request"}

    anthropic_session = _session(
        tmp_path,
        provider="anthropic",
        model="claude-sonnet-4.5",
        base_url="https://api.anthropic.com/v1",
        role_prompt="Act as the Critic.",
        compact_summary="Critic compact summary.",
    )
    anthropic = provider_transport_request(anthropic_session, _tools(anthropic_session))

    assert anthropic.protocol is ProviderApiProtocol.ANTHROPIC_MESSAGES
    assert "Act as the Critic." in anthropic.payload["system"]
    assert "Critic compact summary." in anthropic.payload["system"]
    assert anthropic.payload["messages"] == [{"role": "user", "content": "latest request"}]


def test_provider_transport_converts_context_for_gemini(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        provider="google-gemini-cli",
        model="gemini-3-pro-preview",
        base_url="https://generativelanguage.googleapis.com",
        role_prompt="Act as the researcher.",
        messages=[{"role": "user", "content": "look up source"}, {"role": "assistant", "content": "source noted"}],
    )

    request = provider_transport_request(session, _tools(session))

    assert request.protocol is ProviderApiProtocol.GEMINI_GENERATE_CONTENT
    assert "Act as the researcher." in request.payload["systemInstruction"]["parts"][0]["text"]
    assert request.payload["contents"] == [
        {"role": "user", "parts": [{"text": "look up source"}]},
        {"role": "model", "parts": [{"text": "source noted"}]},
        {"role": "user", "parts": [{"text": "latest request"}]},
    ]


def _session(
    tmp_path: Path,
    *,
    provider: str = "oauth_gateway",
    model: str = "gpt-5.5",
    base_url: str = "https://model-gateway.example/v1",
    role_prompt: str = "",
    compact_summary: str = "",
    workflow_state: dict[str, object] | None = None,
    skills: tuple[str, ...] = (),
    ledger_facts: list[dict[str, object]] | None = None,
    messages: list[dict[str, object]] | None = None,
    tool_history: list[dict[str, object]] | None = None,
) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": provider,
            "model": model,
            "reasoning_effort": "high",
            "base_url": base_url,
            "auth_mode": "gateway",
            "api_key_env": "MODEL_GATEWAY_TOKEN",
        }
    )
    return AsaAgentSession(
        run_id="context-assembler-test",
        session_id="context-session",
        agent_id="orchestrator",
        user_goal="latest request",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
        role_prompt=role_prompt,
        compact_summary=compact_summary,
        workflow_state=dict(workflow_state or {}),
        skills=skills,
        ledger_facts=list(ledger_facts or []),
        messages=list(messages or []),
        tool_history=list(tool_history or []),
    )


def _tools(session: AsaAgentSession) -> tuple[dict[str, object], ...]:
    return tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True)
