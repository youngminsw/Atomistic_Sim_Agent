from __future__ import annotations

import json
from pathlib import Path

from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_transport import provider_transport_request
from sim_agent.cli.tui_state import initial_state
from sim_agent.llm_endpoints import ModelProviderConfig


def test_skill_invoke_tool_runs_registered_skill_and_writes_artifact(tmp_path: Path) -> None:
    state = initial_state(tmp_path)
    registry = default_tool_registry()

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="skill_invoke",
            arguments={
                "skill_id": "qa_physics_and_runtime_evidence",
                "payload": {
                    "request_id": "qa-skill-test",
                    "agent_run_ledger": "ledger.json",
                    "quality_gates": ["provider", "session", "tool"],
                },
            },
            run_id="skill-run",
            session_id=state.session_id,
        ),
        registry,
        state.session_dir,
    )

    assert result.status == "ready"
    assert result.output["skill_id"] == "qa_physics_and_runtime_evidence"
    assert result.output["execution_status"] == "adapter_contract_ready"
    artifact = state.session_dir / "skill_invocations" / "qa-skill-test" / "qa_agent" / "qa_physics_and_runtime_evidence.json"
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == "ready"


def test_skill_invoke_tool_blocks_unknown_skill(tmp_path: Path) -> None:
    state = initial_state(tmp_path)

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="skill_invoke",
            arguments={"skill_id": "missing_skill"},
            run_id="skill-run",
            session_id=state.session_id,
        ),
        default_tool_registry(),
        state.session_dir,
    )

    assert result.status == "blocked"
    assert result.blocker == "unknown_skill"


def test_workflow_start_tool_writes_resumable_workflow_ledger(tmp_path: Path) -> None:
    state = initial_state(tmp_path)

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="workflow_start",
            arguments={
                "workflow_id": "ralplan",
                "payload": {
                    "request_id": "workflow-test",
                    "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
                },
            },
            run_id="workflow-run",
            session_id=state.session_id,
        ),
        default_tool_registry(),
        state.session_dir,
    )

    assert result.status == "ready"
    assert result.output["workflow_id"] == "ralplan"
    assert result.output["resumable"] is True
    assert result.output["gate_status"] == "passed"
    assert (state.session_dir / result.output["ledger_ref"]).is_file()


def test_provider_payload_exposes_skill_and_workflow_tools(tmp_path: Path) -> None:
    session = _session(tmp_path)

    request = provider_transport_request(
        session,
        tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True),
    )

    tools = {tool["name"]: tool for tool in request.payload["tools"]}
    assert "skill_invoke" in tools
    assert "workflow_start" in tools
    assert "qa_physics_and_runtime_evidence" in tools["skill_invoke"]["parameters"]["properties"]["skill_id"]["enum"]
    assert "ultragoal" in tools["workflow_start"]["parameters"]["properties"]["workflow_id"]["enum"]


def _session(tmp_path: Path) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://api.openai.com/v1",
            "auth_mode": "api_key",
            "api_key_env": "OPENAI_API_KEY",
        }
    )
    return AsaAgentSession(
        run_id="skill-workflow-test",
        session_id="skill-workflow-session",
        agent_id="orchestrator",
        user_goal="Invoke a workflow or skill.",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )
