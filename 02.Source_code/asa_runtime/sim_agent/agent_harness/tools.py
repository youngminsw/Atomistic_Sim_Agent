from __future__ import annotations

from typing import Final

from sim_agent.agent_runtime.agent_specs import SUBAGENT_PRESETS
from sim_agent.agents_sdk_runtime.skill_registry import agent_skill_contracts
from sim_agent.agents_sdk_runtime.workflow_harness_types import workflow_harness_catalog

from .agent_runtime_tools import (
    execute_agent_message,
    execute_handoff_task,
    execute_skill_invoke,
    execute_subagent_control,
    execute_subagent_inspect,
    execute_subagent_task,
)
from .workflow_runtime_tools import (
    execute_workflow_gate_response,
    execute_workflow_goal,
    execute_workflow_start,
)
from .tool_policy import DEFAULT_RUNTIME_TOOL_POLICY
from .tool_execution import (
    execute_artifact_write,
    execute_bash_process,
    execute_custom_tool_register,
    execute_file_edit,
    execute_file_read,
    execute_file_search,
    execute_file_write,
    execute_graphdb_dry_run,
    execute_mcp_call_tool,
    execute_mcp_list_tools,
    execute_runtime_tool,
)
from .tool_types import (
    RuntimeToolCall,
    RuntimeToolError,
    RuntimeToolExecutor,
    RuntimeToolResult,
    ToolDefinition,
    ToolRegistry,
)


def default_tool_registry() -> ToolRegistry:
    subagent_preset_enum = list(SUBAGENT_PRESETS)
    subagent_control_action_enum = ["list", "progress", "await", "cancel", "pause", "resume", "steer", "restart"]
    skill_id_enum = [contract.skill_id for contract in agent_skill_contracts()]
    workflow_id_enum = [workflow.workflow_id for workflow in workflow_harness_catalog()]
    return ToolRegistry(
        tools=(
            ToolDefinition(
                "bash_process",
                "process",
                family="process",
                safety="workspace_write",
                approval_required=False,
                executable=True,
                policy_id=DEFAULT_RUNTIME_TOOL_POLICY.policy_id,
                policy_summary=DEFAULT_RUNTIME_TOOL_POLICY.process_policy_summary,
                parameters=_object_schema({"argv": {"type": "array", "items": {"type": "string"}}}, ("argv",)),
                side_effect_class="workspace_process",
                concurrency_policy="exclusive",
                executor=execute_bash_process,
            ),
            ToolDefinition(
                "shell_command",
                "process",
                family="process",
                safety="workspace_write",
                approval_required=False,
                executable=True,
                policy_id=DEFAULT_RUNTIME_TOOL_POLICY.policy_id,
                policy_summary=DEFAULT_RUNTIME_TOOL_POLICY.process_policy_summary,
                parameters=_object_schema({"argv": {"type": "array", "items": {"type": "string"}}}, ("argv",)),
                side_effect_class="workspace_process",
                concurrency_policy="exclusive",
                executor=execute_bash_process,
            ),
            ToolDefinition(
                "artifact_write",
                "evidence",
                family="artifact",
                safety="workspace_write",
                approval_required=False,
                executable=True,
                policy_id="artifact-workspace-confined-v1",
                policy_summary="session_artifacts_subtree_only",
                parameters=_object_schema(
                    {
                        "relative_path": {"type": "string", "description": "Relative artifact path inside this ASA session."},
                        "content": {"type": "string", "description": "Evidence text to write."},
                    },
                    ("relative_path", "content"),
                ),
                side_effect_class="session_artifact_write",
                executor=execute_artifact_write,
            ),
            ToolDefinition(
                "file_read",
                "filesystem",
                family="file",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="session-workspace-file-v1",
                policy_summary="session_workspace_subtree_only",
                parameters=_object_schema(
                    {
                        "relative_path": {"type": "string"},
                        "max_bytes": {"type": "integer"},
                    },
                    ("relative_path",),
                ),
                side_effect_class="none",
                executor=execute_file_read,
            ),
            ToolDefinition(
                "file_write",
                "filesystem",
                family="file",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="session-workspace-file-v1",
                policy_summary="session_workspace_subtree_only",
                parameters=_object_schema(
                    {
                        "relative_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    ("relative_path", "content"),
                ),
                side_effect_class="session_file_write",
                executor=execute_file_write,
            ),
            ToolDefinition(
                "file_search",
                "filesystem",
                family="file",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="session-workspace-file-v1",
                policy_summary="session_workspace_subtree_only",
                parameters=_object_schema(
                    {
                        "query": {"type": "string"},
                        "root": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    ("query",),
                ),
                side_effect_class="none",
                executor=execute_file_search,
            ),
            ToolDefinition(
                "file_edit",
                "filesystem",
                family="file",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="session-workspace-file-v1",
                policy_summary="session_workspace_subtree_only",
                parameters=_object_schema(
                    {
                        "relative_path": {"type": "string"},
                        "search": {"type": "string"},
                        "replace": {"type": "string"},
                        "expected_replacements": {"type": "integer"},
                    },
                    ("relative_path", "search", "replace"),
                ),
                side_effect_class="session_file_write",
                executor=execute_file_edit,
            ),
            ToolDefinition(
                "graphdb_dry_run",
                "graphdb",
                family="graphdb",
                safety="dry_run",
                approval_required=False,
                executable=True,
                policy_id="graphdb-dry-run-v1",
                policy_summary="writes_disabled",
                parameters=_object_schema({"database_name": {"type": "string"}}, ("database_name",)),
                side_effect_class="dry_run",
                executor=execute_graphdb_dry_run,
            ),
            ToolDefinition(
                "mcp_list_tools",
                "mcp",
                family="mcp",
                safety="read_only",
                approval_required=False,
                executable=True,
                policy_id="mcp-configured-list-v1",
                policy_summary="configured_mcp_server_tool_listing",
                parameters=_object_schema({"server_name": {"type": "string"}}, ()),
                side_effect_class="none",
                executor=execute_mcp_list_tools,
            ),
            ToolDefinition(
                "mcp_call_tool",
                "mcp",
                family="mcp",
                safety="configured_external_tool",
                approval_required=False,
                executable=True,
                policy_id="mcp-configured-call-v1",
                policy_summary="configured_mcp_json_rpc_initialize_list_call",
                parameters=_object_schema(
                    {
                        "server_name": {"type": "string"},
                        "tool_name": {"type": "string"},
                        "arguments": {"type": "object", "additionalProperties": True},
                    },
                    ("server_name", "tool_name"),
                ),
                side_effect_class="configured_mcp_call",
                concurrency_policy="exclusive",
                executor=execute_mcp_call_tool,
            ),
            ToolDefinition(
                "custom_tool_register",
                "custom_tool",
                family="custom_tool",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="custom-tool-descriptor-v1",
                policy_summary="register_descriptor_only_no_execution",
                parameters=_object_schema(
                    {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters": {"type": "object", "additionalProperties": True},
                    },
                    ("name", "description", "parameters"),
                ),
                side_effect_class="session_custom_tool_descriptor",
                executor=execute_custom_tool_register,
            ),
            ToolDefinition(
                "agent_message",
                "agent_bus",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="agent-message-bus-v1",
                policy_summary="session_local_append_only_bus",
                parameters=_object_schema(
                    {
                        "action": {"type": "string", "enum": ["send", "ack", "read", "reply"]},
                        "from_agent": {"type": "string"},
                        "to_agent": {"type": "string"},
                        "content": {"type": "string"},
                        "message_id": {"type": "string"},
                        "by_agent": {"type": "string"},
                        "thread_id": {"type": "string"},
                    },
                    ("action",),
                ),
                side_effect_class="session_message_bus",
                executor=execute_agent_message,
            ),
            ToolDefinition(
                "handoff_task",
                "agent_handoff",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="handoff-task-v1",
                policy_summary="bounded_target_agent_session_handoff",
                parameters=_object_schema(
                    {
                        "target_agent": {"type": "string"},
                        "task": {"type": "string"},
                        "from_agent": {"type": "string"},
                        "task_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                    },
                    ("target_agent", "task"),
                ),
                side_effect_class="session_handoff",
                executor=execute_handoff_task,
            ),
            ToolDefinition(
                "subagent_task",
                "agent_subagent",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="subagent-task-v1",
                policy_summary="bounded_clean_room_non_persistent_subagent_run",
                parameters=_object_schema(
                    {
                        "caller_agent": {"type": "string"},
                        "preset": {"type": "string", "enum": subagent_preset_enum},
                        "task": {"type": "string"},
                        "task_id": {"type": "string"},
                        "depth": {"type": "integer"},
                    },
                    ("caller_agent", "preset", "task"),
                ),
                side_effect_class="bounded_subagent_job",
                executor=execute_subagent_task,
            ),
            ToolDefinition(
                "subagent_inspect",
                "agent_subagent",
                family="agent_runtime",
                safety="read_only",
                approval_required=False,
                executable=True,
                policy_id="subagent-inspect-v1",
                policy_summary="read_only_bounded_subagent_inspection",
                parameters=_object_schema(
                    {
                        "caller_agent": {"type": "string"},
                        "preset": {"type": "string", "enum": subagent_preset_enum},
                        "subagent_id": {"type": "string"},
                    },
                    ("caller_agent", "preset", "subagent_id"),
                ),
                side_effect_class="none",
                executor=execute_subagent_inspect,
            ),
            ToolDefinition(
                "subagent_control",
                "agent_subagent",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="subagent-control-v1",
                policy_summary="session_local_bounded_subagent_lifecycle_control",
                parameters=_object_schema(
                    {
                        "action": {
                            "type": "string",
                            "enum": subagent_control_action_enum,
                        },
                        "caller_agent": {"type": "string"},
                        "preset": {"type": "string", "enum": subagent_preset_enum},
                        "subagent_id": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    ("action", "caller_agent"),
                ),
                side_effect_class="bounded_subagent_control",
                executor=execute_subagent_control,
            ),
            ToolDefinition(
                "skill_invoke",
                "agent_skill",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="skill-invoke-v1",
                policy_summary="session_local_skill_adapter_preflight",
                parameters=_object_schema(
                    {
                        "skill_id": {
                            "type": "string",
                            "enum": skill_id_enum,
                        },
                        "payload": {"type": "object", "additionalProperties": True},
                    },
                    ("skill_id",),
                ),
                side_effect_class="session_skill_artifact",
                executor=execute_skill_invoke,
            ),
            ToolDefinition(
                "workflow_start",
                "agent_workflow",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="workflow-start-v1",
                policy_summary="session_local_workflow_harness_checkpoint",
                parameters=_object_schema(
                    {
                        "workflow_id": {
                            "type": "string",
                            "enum": workflow_id_enum,
                        },
                        "owner_agent_id": {"type": "string"},
                        "target_agent_id": {"type": "string"},
                        "goal_id": {"type": "string"},
                        "payload": {"type": "object", "additionalProperties": True},
                    },
                    ("workflow_id",),
                ),
                side_effect_class="session_workflow_ledger",
                executor=execute_workflow_start,
            ),
            ToolDefinition(
                "workflow_gate_response",
                "agent_workflow",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="workflow-gate-response-v1",
                policy_summary="session_local_workflow_gate_response",
                parameters=_object_schema(
                    {
                        "workflow_id": {"type": "string"},
                        "gate_id": {"type": "string"},
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "integer"},
                                {"type": "boolean"},
                                {"type": "object", "additionalProperties": True},
                                {"type": "array", "items": {}},
                                {"type": "null"},
                            ]
                        },
                    },
                    ("workflow_id", "gate_id", "value"),
                ),
                side_effect_class="session_workflow_ledger",
                executor=execute_workflow_gate_response,
            ),
            ToolDefinition(
                "workflow_goal",
                "agent_workflow",
                family="agent_runtime",
                safety="session_local",
                approval_required=False,
                executable=True,
                policy_id="workflow-goal-v1",
                policy_summary="session_local_workflow_goal_state",
                parameters=_object_schema(
                    {
                        "operation": {
                            "type": "string",
                            "enum": ["create", "get", "resume", "pause", "drop", "complete"],
                        },
                        "workflow_id": {"type": "string"},
                        "goal_id": {"type": "string"},
                        "owner_agent_id": {"type": "string"},
                        "target_agent_id": {"type": "string"},
                        "objective": {"type": "string"},
                    },
                    ("operation", "workflow_id", "goal_id", "owner_agent_id", "target_agent_id"),
                ),
                side_effect_class="session_workflow_goal",
                executor=execute_workflow_goal,
            ),
            ToolDefinition("validate_simulation_request", "schema"),
            ToolDefinition("geometry_ingestion", "geometry"),
            ToolDefinition("md_campaign_planning", "md"),
            ToolDefinition("surrogate_status", "mdn"),
            ToolDefinition("feature_transport", "transport"),
            ToolDefinition("level_set_evolution", "profile"),
            ToolDefinition("compute_routing", "compute"),
            ToolDefinition("artifact_manifest", "evidence"),
            ToolDefinition("literature_registry", "graphdb"),
            ToolDefinition("research_source_lookup", "graphdb"),
            ToolDefinition("source_graph_import_bundle", "graphdb"),
            ToolDefinition("graphdb_ingest_report", "graphdb"),
            ToolDefinition("ui_run_status", "html_ui"),
        )
    )


def _object_schema(properties: dict[str, object], required: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": True,
    }


COMMON_DOMAIN_TOOL_NAMES: Final = frozenset(
    (
        "artifact_write",
        "file_read",
        "file_write",
        "file_search",
        "file_edit",
        "mcp_list_tools",
        "mcp_call_tool",
        "subagent_task",
        "subagent_inspect",
        "subagent_control",
        "workflow_start",
        "workflow_gate_response",
        "workflow_goal",
    )
)


DOMAIN_AGENT_TOOL_NAMES: Final[dict[str, frozenset[str]]] = {
    "md_agent": frozenset(
        (
            "bash_process",
            "shell_command",
            "md_campaign_planning",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
    "ml_agent": frozenset(
        (
            "surrogate_status",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
    "feature_scale_agent": frozenset(
        (
            "feature_transport",
            "level_set_evolution",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
    "research_agent": frozenset(
        (
            "graphdb_dry_run",
            "literature_registry",
            "mcp_list_tools",
            "custom_tool_register",
            "research_source_lookup",
            "source_graph_import_bundle",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
    "qa_agent": frozenset(
        (
            "artifact_manifest",
            "mcp_list_tools",
            "custom_tool_register",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
    "orchestrator": frozenset(
        (
            "artifact_manifest",
            "bash_process",
            "shell_command",
            "custom_tool_register",
            "handoff_task",
            "mcp_list_tools",
            "skill_invoke",
            "subagent_control",
            "workflow_start",
            "workflow_gate_response",
            "workflow_goal",
        )
    )
    | COMMON_DOMAIN_TOOL_NAMES,
}


def tool_registry_for_agent(agent_id: str) -> ToolRegistry:
    tool_names = DOMAIN_AGENT_TOOL_NAMES.get(agent_id)
    if tool_names is None:
        raise RuntimeToolError("unknown_agent_id")
    return ToolRegistry(tuple(tool for tool in default_tool_registry().tools if tool.name in tool_names))


__all__ = [
    "RuntimeToolCall",
    "RuntimeToolError",
    "RuntimeToolExecutor",
    "RuntimeToolResult",
    "ToolDefinition",
    "ToolRegistry",
    "default_tool_registry",
    "execute_runtime_tool",
    "tool_registry_for_agent",
]
