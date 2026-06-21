from __future__ import annotations

from .agent_runtime_tools import (
    execute_agent_message,
    execute_handoff_task,
    execute_subagent_inspect,
    execute_subagent_task,
)
from .tool_policy import DEFAULT_RUNTIME_TOOL_POLICY
from .tool_execution import (
    execute_artifact_write,
    execute_bash_process,
    execute_graphdb_dry_run,
    execute_runtime_tool,
)
from .tool_types import RuntimeToolCall, RuntimeToolError, RuntimeToolExecutor, RuntimeToolResult, ToolDefinition, ToolRegistry


def default_tool_registry() -> ToolRegistry:
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
                executor=execute_artifact_write,
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
                executor=execute_graphdb_dry_run,
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
                executor=execute_subagent_inspect,
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


__all__ = [
    "RuntimeToolCall",
    "RuntimeToolError",
    "RuntimeToolExecutor",
    "RuntimeToolResult",
    "ToolDefinition",
    "ToolRegistry",
    "default_tool_registry",
    "execute_runtime_tool",
]
