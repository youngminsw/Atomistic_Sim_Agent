from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.agent_runtime import (
    AutoCompactionPolicy,
    CompactionRequest,
    append_agent_message,
    compact_agent_session,
    load_agent_registry,
    replay_agent_compaction,
)
from sim_agent.agent_runtime.agent_registry import AgentSessionHandle
from sim_agent.agent_runtime.agent_specs import resolve_subagent_preset
from sim_agent.agent_runtime.compaction_semantic import SemanticSummaryRequest, SemanticSummaryResult
from sim_agent.agent_runtime.global_session_types import GlobalSessionModel
from sim_agent.agent_runtime.live_agent_context import live_turn_project_guidance
from sim_agent.agent_runtime.subagent_loop import _subagent_agent_session
from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession, assemble_provider_context
from sim_agent.agents_sdk_runtime.markdown_skills import MarkdownSkillSpec, skill_context_message
from sim_agent.agents_sdk_runtime.prompt_assets import (
    load_common_system_prompt,
    load_domain_role_prompt,
    load_subagent_role_prompt,
    load_workflow_policy_prompt,
)
from sim_agent.agents_sdk_runtime.provider_transport import ProviderApiProtocol, provider_transport_request
from sim_agent.cli.tui_state import initial_state
from sim_agent.llm_endpoints import ModelProviderConfig


@dataclass(slots=True)
class RecordingSummarizer:
    result: SemanticSummaryResult
    requests: list[SemanticSummaryRequest]

    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        self.requests.append(request)
        return self.result


def test_provider_context_assembles_role_compaction_transcript_and_tool_history(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        role_prompt="Act as the MD domain agent.",
        workflow_policy="Validate request, plan tools, then execute only receipt-backed actions.",
        project_guidance="Project guidance: keep ASA runtime state under the current WSL session.",
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

    assert context.layer_kinds() == (
        "system_policy",
        "workflow_policy",
        "domain_role",
        "project_guidance",
        "compact_summary",
        "skills",
        "workflow_state",
        "ledger_facts",
        "tool_history",
    )
    assert context.layers_json()[2]["kind"] == "domain_role"
    assert "Act as the MD domain agent." in context.instructions
    assert "Validate request, plan tools" in context.instructions
    assert "current WSL session" in context.instructions
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


def test_provider_context_uses_file_backed_common_and_workflow_policy(tmp_path: Path) -> None:
    session = _session(tmp_path)

    context = assemble_provider_context(session)
    layers = context.layers_json()

    assert layers[0]["kind"] == "system_policy"
    assert layers[0]["source"] == "asa.prompts.system.common_system"
    assert load_common_system_prompt() in layers[0]["content"]
    assert "<identity>" in layers[0]["content"]
    assert "<completion-contract>" in layers[0]["content"]
    assert layers[1]["kind"] == "workflow_policy"
    assert layers[1]["source"] == "asa.prompts.system.workflow_policy"
    assert load_workflow_policy_prompt() in layers[1]["content"]


def test_common_system_prompt_contains_asa_constitution_without_domain_manual() -> None:
    common = load_common_system_prompt()

    assert "Python-native" in common
    assert "semiconductor" in common
    assert "plasma" in common or "dry etching" in common
    assert "pattern evolution" in common
    assert "Slash commands activate skills or workflows" in common
    assert "Direct @agent messaging is direct persistent agent messaging" in common
    assert "registered known agent handles" in common
    assert "Unknown @agent targets are blocked" in common
    assert "arbitrary dynamic agent creation" in common
    assert "Conversation messages are provider input after prompt layers" in common
    assert "Conversation messages are not a PromptLayer" in common
    assert "system policy, workflow policy, role prompt, project guidance, compact summary, skills, workflow state, ledger facts, and tool history" in common

    for phrase in (
        "LAMMPS-oriented atomistic simulation campaigns",
        "MDN or surrogate training",
        "Level-Set/profile evolution inputs",
        "Neo4j write approval boundaries",
    ):
        assert phrase not in common


def test_prompt_assets_do_not_contain_disallowed_runtime_branding() -> None:
    prompt_texts = [
        load_common_system_prompt(),
        load_workflow_policy_prompt(),
        *(load_domain_role_prompt(agent) for agent in ("orchestrator", "md_agent", "ml_agent", "feature_scale_agent", "research_agent", "qa_agent")),
        *(load_subagent_role_prompt(role) for role in ("planner", "architect", "critic", "executor", "verifier")),
    ]

    for text in prompt_texts:
        assert "GJC" not in text
        assert "Gajae" not in text
        assert ".gjc" not in text
        assert "gjc " not in text
        assert "gjc-" not in text


def test_domain_role_prompts_own_detailed_methodology() -> None:
    md = load_domain_role_prompt("md_agent")
    ml = load_domain_role_prompt("ml_agent")
    feature = load_domain_role_prompt("feature_scale_agent")
    research = load_domain_role_prompt("research_agent")
    qa = load_domain_role_prompt("qa_agent")

    assert "force-field" in md
    assert "LAMMPS" in md
    assert "trajectory" in md
    assert "event" in md
    assert "coverage" in ml
    assert "surrogate" in ml or "MDN" in ml
    assert "calibration" in ml or "uncertainty" in ml
    assert "KMC" in feature
    assert "transport" in feature
    assert "Level-Set" in feature
    assert "profile evolution" in feature
    assert "provenance" in research
    assert "GraphDB/MCP" in research
    assert "evidence audit" in qa or "Audit runtime evidence" in qa
    assert "hard blocker" in qa or "Reject final completion" in qa


def test_subagent_role_prompts_are_global_bounded_roles() -> None:
    planner = load_subagent_role_prompt("planner")
    architect = load_subagent_role_prompt("architect")
    critic = load_subagent_role_prompt("critic")
    executor = load_subagent_role_prompt("executor")
    verifier = load_subagent_role_prompt("verifier")

    for text in (planner, architect, critic, executor, verifier):
        assert "bounded" in text
        assert "persistent domain agent" in text

    assert "code, design, workflow, tool safety, evidence quality, and scientific validity" in critic
    assert "Use only the tools assigned" in executor
    assert "persistent domain identity" not in executor


def test_domain_role_prompt_is_loaded_from_markdown_asset(tmp_path: Path) -> None:
    session = _session(tmp_path, role_prompt=load_domain_role_prompt("md_agent"))

    context = assemble_provider_context(session)

    assert "domain_role" in context.layer_kinds()
    assert "<identity>" in context.instructions
    assert "You are the MD Agent" in context.instructions
    assert "force-field" in context.instructions


def test_provider_context_drops_invalid_messages_and_skips_unserializable_layers(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        workflow_state={"bad": object()},
        ledger_facts=[{"bad": object()}],
        messages=[
            {"role": "system", "content": "USER_SYSTEM_HIDDEN_MARKER"},
            {"role": "tool", "content": "not provider-visible"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "latest request"},
        ],
    )

    context = assemble_provider_context(session)

    assert context.openai_responses_input() == [{"role": "user", "content": "latest request"}]
    assert "workflow_state" not in context.layer_kinds()
    assert "ledger_facts" not in context.layer_kinds()
    assert "USER_SYSTEM_HIDDEN_MARKER" not in context.instructions


def test_provider_context_injects_markdown_skill_context_as_skill_layer(tmp_path: Path) -> None:
    spec = MarkdownSkillSpec(
        name="project-plan",
        command="/project-plan",
        agent_id="qa_agent",
        summary="Project plan skill",
        path=tmp_path / "project-plan.md",
        body="Use the project plan skill body for QA routing.",
    )
    session = _session(
        tmp_path,
        messages=[
            {"role": "system", "content": "generic system note that must stay hidden"},
            {"role": "system", "content": skill_context_message(spec)},
            {"role": "system", "content": "ASA_SKILL_CONTEXT_V1 malformed-no-body"},
            {"role": "user", "content": "latest request"},
        ],
    )

    context = assemble_provider_context(session)

    assert "skills" in context.layer_kinds()
    assert "Skill: project-plan" in context.instructions
    assert "Use the project plan skill body" in context.instructions
    assert "generic system note" not in context.instructions
    assert "malformed-no-body" not in context.instructions
    assert context.openai_responses_input() == [{"role": "user", "content": "latest request"}]


def test_subagent_preset_role_is_assembled_as_role_layer_not_user_goal(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    preset = resolve_subagent_preset("critic")
    session = _subagent_agent_session(handle, preset, "review-code", "Review code and evidence.", 1, tmp_path / "child")

    context = assemble_provider_context(session)
    user_turn = context.openai_responses_input()[-1]["content"]

    assert "subagent_role" in context.layer_kinds()
    assert "Challenge ASA work products" in context.instructions
    assert "Review code, design, workflow" in context.instructions
    assert "Role:" not in user_turn
    assert "Scope:" not in user_turn
    assert "Task: Review code and evidence." in user_turn


def test_subagent_caller_context_uses_compacted_provider_visible_tail(tmp_path: Path) -> None:
    state = initial_state(tmp_path / "session")
    append_agent_message(state.session_dir, "md_agent", "user", "SUBAGENT_OLD_RAW_SHOULD_NOT_LEAK")
    for index in range(29):
        role = "assistant" if index % 2 else "user"
        append_agent_message(state.session_dir, "md_agent", role, f"SUBAGENT_TAIL_CONTEXT_{index}")
    compact_agent_session(
        state.session_dir,
        CompactionRequest(
            agent_id="md_agent",
            compact_id="compact-md-subagent",
            summary="Subagent caller compact summary.",
        ),
        summarizer=RecordingSummarizer(SemanticSummaryResult(summary="Subagent caller compact summary."), []),
        policy=AutoCompactionPolicy(context_window_tokens=10_000, keep_recent_tokens=96),
    )
    replayed = replay_agent_compaction(state.session_dir, "md_agent")
    handle = load_agent_registry(state.session_dir).handles["md_agent"]
    preset = resolve_subagent_preset("critic")
    session = _subagent_agent_session(handle, preset, "review-compact", "Review compacted caller context.", 1, tmp_path / "child")

    context = assemble_provider_context(session)

    assert replayed.status == "succeeded"
    assert "caller_context" in context.layer_kinds()
    assert "Subagent caller compact summary." in context.instructions
    assert "SUBAGENT_TAIL_CONTEXT_28" in context.instructions
    assert "SUBAGENT_OLD_RAW_SHOULD_NOT_LEAK" not in context.instructions
    assert "Review compacted caller context." in context.openai_responses_input()[-1]["content"]


def test_provider_context_promotes_only_marked_skill_system_messages(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_by_command, skill_context_message

    spec = markdown_skill_by_command("/md")
    assert spec is not None
    session = _session(
        tmp_path,
        messages=[
            {"role": "system", "content": "generic hidden system note"},
            {"role": "system", "content": skill_context_message(spec)},
            {"role": "user", "content": "use md skill"},
        ],
    )

    context = assemble_provider_context(session)

    assert "skills" in context.layer_kinds()
    assert "Skill: md" in context.instructions
    assert "# MD Skill" in context.instructions
    assert "generic hidden system note" not in context.instructions
    assert context.openai_responses_input() == [
        {"role": "user", "content": "use md skill"},
        {"role": "user", "content": "latest request"},
    ]


def test_live_project_guidance_injects_project_and_asa_guidance_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = \"sample\"\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("top-level guidance\n", encoding="utf-8")
    (project / ".asa").mkdir()
    (project / ".asa" / "AGENTS.md").write_text("asa agent guidance\n", encoding="utf-8")
    (project / ".asa" / "SKILLS.md").write_text("asa skill guidance\n", encoding="utf-8")
    monkeypatch.chdir(project)

    guidance = live_turn_project_guidance(_handle(tmp_path))

    assert "top-level guidance" in guidance
    assert "asa agent guidance" in guidance
    assert "asa skill guidance" in guidance


def test_live_project_guidance_ignores_non_utf_guidance_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = \"sample\"\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("readable guidance\n", encoding="utf-8")
    (project / ".asa").mkdir()
    (project / ".asa" / "AGENTS.md").write_bytes(b"\xff\xfe\x00not utf-8")
    monkeypatch.chdir(project)

    guidance = live_turn_project_guidance(_handle(tmp_path))

    assert "readable guidance" in guidance
    assert "not utf-8" not in guidance


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
    role_prompt_kind: str = "domain_role",
    workflow_policy: str = "",
    project_guidance: str = "",
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
        role_prompt_kind=role_prompt_kind,
        workflow_policy=workflow_policy,
        project_guidance=project_guidance,
        compact_summary=compact_summary,
        workflow_state=dict(workflow_state or {}),
        skills=skills,
        ledger_facts=list(ledger_facts or []),
        messages=list(messages or []),
        tool_history=list(tool_history or []),
    )


def _handle(tmp_path: Path) -> AgentSessionHandle:
    model = GlobalSessionModel(
        provider="openai-codex",
        name="gpt-5-codex",
        reasoning_effort="high",
        base_url="https://model-gateway.local/v1",
        auth_mode="gateway",
        api_key_env="MODEL_GATEWAY_TOKEN",
    )
    return AgentSessionHandle(
        agent_id="orchestrator",
        display_name="Orchestrator",
        boundary="test",
        role_prompt="",
        agent_session_id="agent-session",
        session_dir=tmp_path / "session",
        messages_path=tmp_path / "messages.jsonl",
        events_path=tmp_path / "events.jsonl",
        model=model,
        created_at=0.0,
    )


def _tools(session: AsaAgentSession) -> tuple[dict[str, object], ...]:
    return tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True)
