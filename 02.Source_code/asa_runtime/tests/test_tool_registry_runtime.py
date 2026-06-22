from __future__ import annotations

from pathlib import Path

from sim_agent.agent_harness.tools import default_tool_registry
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_transport import provider_transport_request
from sim_agent.llm_endpoints import ModelProviderConfig


def test_executable_tool_definitions_carry_provider_schema_metadata() -> None:
    registry = default_tool_registry()

    artifact = registry.require_tool("artifact_write")
    subagent_control = registry.require_tool("subagent_control")
    unavailable = registry.require_tool("validate_simulation_request")

    assert artifact.parameters is not None
    assert artifact.parameters["required"] == ["relative_path", "content"]
    assert artifact.side_effect_class == "session_artifact_write"
    assert subagent_control.parameters is not None
    assert subagent_control.parameters["properties"]["action"]["enum"] == [
        "list",
        "progress",
        "await",
        "cancel",
        "pause",
        "resume",
        "steer",
    ]
    assert subagent_control.side_effect_class == "bounded_subagent_control"
    assert unavailable.executable is False
    assert unavailable.side_effect_class == "none"


def test_provider_payload_uses_registry_parameters(tmp_path: Path) -> None:
    session = _session(tmp_path)

    request = provider_transport_request(
        session,
        tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True),
    )

    tools = {tool["name"]: tool for tool in request.payload["tools"]}
    assert tools["artifact_write"]["parameters"] == default_tool_registry().require_tool("artifact_write").parameters
    assert tools["subagent_control"]["parameters"] == default_tool_registry().require_tool("subagent_control").parameters
    assert "validate_simulation_request" not in tools


def test_model_visible_tool_schema_projects_runtime_metadata(tmp_path: Path) -> None:
    session = _session(tmp_path)

    schemas = {schema["name"]: schema for schema in session.model_visible_tool_schemas()}

    assert schemas["artifact_write"]["side_effect_class"] == "session_artifact_write"
    assert schemas["artifact_write"]["strict"] is True
    assert schemas["subagent_task"]["owner"] == "runtime"
    assert schemas["subagent_control"]["concurrency_policy"] == "serial"


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
        run_id="tool-registry-test",
        session_id="tool-registry-session",
        agent_id="orchestrator",
        user_goal="Select safe tools.",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=default_tool_registry(),
    )
