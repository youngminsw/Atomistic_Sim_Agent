from __future__ import annotations

from .agent_runtime_tools import (
    execute_agent_message,
    execute_handoff_task,
    execute_skill_invoke,
    execute_subagent_control,
    execute_subagent_inspect,
    execute_subagent_task,
    execute_workflow_start,
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
                        "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
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
                        "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
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
                            "enum": ["list", "progress", "await", "cancel", "pause", "resume", "steer"],
                        },
                        "caller_agent": {"type": "string"},
                        "preset": {"type": "string", "enum": ["planner", "architect", "critic", "executor"]},
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
                            "enum": [
                                "orchestrate_simulation_run",
                                "prepare_and_verify_lammps_md",
                                "train_and_gate_mdn_surrogate",
                                "run_feature_scale_level_set",
                                "research_and_ingest_graphdb_catalog",
                                "qa_physics_and_runtime_evidence",
                            ],
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
                            "enum": ["deep-interview", "ralplan", "ultrawork", "ultraqa", "ultragoal"],
                        },
                        "payload": {"type": "object", "additionalProperties": True},
                    },
                    ("workflow_id",),
                ),
                side_effect_class="session_workflow_ledger",
                executor=execute_workflow_start,
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
