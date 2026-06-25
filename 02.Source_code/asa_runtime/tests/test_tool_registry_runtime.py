from __future__ import annotations

from pathlib import Path

import pytest

from sim_agent.agent_harness.tools import RuntimeToolError, default_tool_registry, tool_registry_for_agent
from sim_agent.agents_sdk_runtime import AsaAgentSession
from sim_agent.agents_sdk_runtime.provider_transport import provider_transport_request
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.schemas._parse import as_mapping, as_sequence


def test_executable_tool_definitions_carry_provider_schema_metadata() -> None:
    registry = default_tool_registry()

    artifact = registry.require_tool("artifact_write")
    file_read = registry.require_tool("file_read")
    file_edit = registry.require_tool("file_edit")
    custom = registry.require_tool("custom_tool_register")
    mcp = registry.require_tool("mcp_list_tools")
    mcp_call = registry.require_tool("mcp_call_tool")
    subagent_control = registry.require_tool("subagent_control")
    unavailable = registry.require_tool("validate_simulation_request")

    assert artifact.parameters is not None
    assert artifact.parameters["required"] == ["relative_path", "content"]
    assert artifact.side_effect_class == "session_artifact_write"
    assert file_read.parameters is not None
    assert file_read.parameters["required"] == ["relative_path"]
    assert file_read.policy_summary == "session_workspace_subtree_only"
    assert file_edit.side_effect_class == "session_file_write"
    assert custom.parameters is not None
    assert custom.parameters["required"] == ["name", "description", "parameters"]
    assert custom.family == "custom_tool"
    assert mcp.family == "mcp"
    assert mcp.side_effect_class == "none"
    assert mcp.policy_summary == "configured_mcp_server_tool_listing"
    assert mcp_call.parameters is not None
    assert mcp_call.parameters["required"] == ["server_name", "tool_name"]
    assert mcp_call.family == "mcp"
    assert mcp_call.side_effect_class == "configured_mcp_call"
    assert mcp_call.policy_summary == "configured_mcp_json_rpc_initialize_list_call"
    assert subagent_control.parameters is not None
    subagent_control_parameters = subagent_control.parameters
    assert subagent_control_parameters is not None
    properties = as_mapping(subagent_control_parameters["properties"], "properties")
    action = as_mapping(properties["action"], "action")
    assert list(as_sequence(action["enum"], "enum")) == [
        "list",
        "progress",
        "await",
        "cancel",
        "pause",
        "resume",
        "steer",
        "restart",
    ]
    preset = as_mapping(
        as_mapping(default_tool_registry().require_tool("subagent_task").parameters or {}, "subagent_task.parameters")[
            "properties"
        ]["preset"],
        "subagent_task.preset",
    )
    assert list(as_sequence(preset["enum"], "enum")) == ["planner", "architect", "critic", "executor", "verifier"]
    assert subagent_control.side_effect_class == "bounded_subagent_control"
    assert unavailable.executable is False
    assert unavailable.side_effect_class == "none"


def test_provider_payload_uses_registry_parameters(tmp_path: Path) -> None:
    session = _session(tmp_path)

    request = provider_transport_request(
        session,
        tuple(schema for schema in session.model_visible_tool_schemas() if schema.get("executable") is True),
    )

    tools = {
        as_mapping(tool, "tool")["name"]: as_mapping(tool, "tool")
        for tool in as_sequence(request.payload["tools"], "tools")
    }
    assert tools["artifact_write"]["parameters"] == default_tool_registry().require_tool("artifact_write").parameters
    assert tools["file_read"]["parameters"] == default_tool_registry().require_tool("file_read").parameters
    assert tools["file_edit"]["parameters"] == default_tool_registry().require_tool("file_edit").parameters
    assert tools["custom_tool_register"]["parameters"] == default_tool_registry().require_tool("custom_tool_register").parameters
    assert tools["subagent_control"]["parameters"] == default_tool_registry().require_tool("subagent_control").parameters
    assert "validate_simulation_request" not in tools


def test_model_visible_tool_schema_projects_runtime_metadata(tmp_path: Path) -> None:
    session = _session(tmp_path)

    schemas = {schema["name"]: schema for schema in session.model_visible_tool_schemas()}

    assert schemas["artifact_write"]["side_effect_class"] == "session_artifact_write"
    assert schemas["artifact_write"]["strict"] is True
    assert schemas["file_write"]["family"] == "file"
    assert schemas["file_write"]["side_effect_class"] == "session_file_write"
    assert schemas["mcp_list_tools"]["policy_id"] == "mcp-configured-list-v1"
    assert schemas["custom_tool_register"]["side_effect_class"] == "session_custom_tool_descriptor"
    assert schemas["subagent_task"]["owner"] == "runtime"
    assert schemas["subagent_control"]["concurrency_policy"] == "serial"


def test_domain_agent_tool_registry_partitions_tools_by_role() -> None:
    md_tools = tool_registry_for_agent("md_agent").tool_names
    research_tools = tool_registry_for_agent("research_agent").tool_names
    qa_tools = tool_registry_for_agent("qa_agent").tool_names

    for agent_id in ("md_agent", "ml_agent", "feature_scale_agent", "research_agent", "qa_agent", "orchestrator"):
        agent_tools = tool_registry_for_agent(agent_id).tool_names
        assert "mcp_list_tools" in agent_tools
        assert "mcp_call_tool" in agent_tools
        assert "subagent_task" in agent_tools
        assert "subagent_inspect" in agent_tools
        assert "subagent_control" in agent_tools
    assert "md_campaign_planning" in md_tools
    assert "file_read" in md_tools
    assert "shell_command" in md_tools
    assert "research_source_lookup" not in md_tools
    assert "graphdb_dry_run" in research_tools
    assert "mcp_list_tools" in research_tools
    assert "custom_tool_register" in research_tools
    assert "research_source_lookup" in research_tools
    assert "artifact_manifest" in qa_tools
    assert "file_edit" in qa_tools
    assert "subagent_task" in qa_tools


def test_domain_agent_tool_registry_rejects_unknown_role_ids() -> None:
    with pytest.raises(RuntimeToolError, match="unknown_agent_id"):
        tool_registry_for_agent("graph_worker")

    with pytest.raises(RuntimeToolError, match="unknown_agent_id"):
        tool_registry_for_agent("surrogate_worker")


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
