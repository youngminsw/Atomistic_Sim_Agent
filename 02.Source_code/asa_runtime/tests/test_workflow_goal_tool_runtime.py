from __future__ import annotations

import json
from pathlib import Path


def test_workflow_goal_operations_persist_state_machine(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import operate_workflow_goal

    create = operate_workflow_goal(
        tmp_path,
        {
            "operation": "create",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "md_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-md",
            "objective": "Run MD parity workflow",
        },
    )
    pause = operate_workflow_goal(
        tmp_path,
        {
            "operation": "pause",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "md_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-md",
        },
    )
    resume = operate_workflow_goal(
        tmp_path,
        {
            "operation": "resume",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "md_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-md",
        },
    )
    complete = operate_workflow_goal(
        tmp_path,
        {
            "operation": "complete",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "md_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-md",
        },
    )
    get = operate_workflow_goal(
        tmp_path,
        {
            "operation": "get",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "md_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-md",
        },
    )
    goal_path = tmp_path / "ultragoal" / "goals" / "goal-md.json"
    goal = json.loads(goal_path.read_text(encoding="utf-8"))

    assert create.status == "accepted"
    assert create.state == "active"
    assert pause.state == "paused"
    assert resume.state == "active"
    assert complete.state == "complete"
    assert get.state == "complete"
    assert get.goal is not None
    assert goal["schema_version"] == "workflow_goal_v1"
    assert goal["objective"] == "Run MD parity workflow"
    assert [event["operation"] for event in goal["history"]] == ["create", "pause", "resume", "complete"]


def test_workflow_goal_operations_enforce_authority_and_known_actions(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import operate_workflow_goal

    peer = operate_workflow_goal(
        tmp_path,
        {
            "operation": "create",
            "actor_agent_id": "md_agent",
            "owner_agent_id": "md_agent",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-peer",
        },
    )
    to_orchestrator = operate_workflow_goal(
        tmp_path,
        {
            "operation": "create",
            "actor_agent_id": "qa_agent",
            "owner_agent_id": "qa_agent",
            "target_agent_id": "orchestrator",
            "workflow_id": "ultragoal",
            "goal_id": "goal-orchestrator",
        },
    )
    invalid = operate_workflow_goal(
        tmp_path,
        {
            "operation": "archive",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultragoal",
            "goal_id": "goal-invalid",
        },
    )
    missing = operate_workflow_goal(
        tmp_path,
        {
            "operation": "get",
            "actor_agent_id": "orchestrator",
            "owner_agent_id": "orchestrator",
            "target_agent_id": "qa_agent",
            "workflow_id": "ultragoal",
            "goal_id": "missing",
        },
    )

    assert peer.status == "blocked"
    assert peer.blockers == ("workflow_authority_peer_denied",)
    assert to_orchestrator.status == "blocked"
    assert to_orchestrator.blockers == ("workflow_authority_orchestrator_denied",)
    assert invalid.status == "blocked"
    assert invalid.blockers == ("workflow_goal_unknown_operation",)
    assert missing.status == "blocked"
    assert missing.blockers == ("workflow_goal_unknown",)


def test_workflow_goal_tool_is_model_visible_and_uses_caller_identity(tmp_path: Path) -> None:
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool, tool_registry_for_agent
    from sim_agent.cli.tui_state import initial_state

    state = initial_state(tmp_path)
    registry = default_tool_registry()
    tools = {tool.name: tool for tool in registry.tools}
    goal_tool = tools["workflow_goal"]

    assert goal_tool.parameters["properties"]["operation"]["enum"] == [
        "create",
        "get",
        "resume",
        "pause",
        "drop",
        "complete",
    ]
    assert "actor_agent_id" not in goal_tool.parameters["properties"]

    domain_registry = tool_registry_for_agent("md_agent")
    assert "workflow_goal" in domain_registry.tool_names

    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="workflow_goal",
            arguments={
                "operation": "create",
                "workflow_id": "ultragoal",
                "goal_id": "goal-md-self",
                "owner_agent_id": "md_agent",
                "target_agent_id": "md_agent",
                "objective": "Self-owned MD goal",
            },
            run_id="workflow-goal-run",
            session_id=state.session_id,
            caller_agent_id="md_agent",
        ),
        registry,
        state.session_dir,
    )

    assert result.status == "accepted"
    assert result.output["actor_agent_id"] == "md_agent"
    assert result.output["state"] == "active"
    assert (state.session_dir / "workflows" / "ultragoal" / "goals" / "goal-md-self.json").is_file()
