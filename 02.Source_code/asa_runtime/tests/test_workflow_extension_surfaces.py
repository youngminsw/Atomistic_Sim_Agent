from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_visual_qa_and_ultraresearch_are_runtime_workflows_with_artifacts(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke, workflow_harness_catalog

    workflow_ids = {workflow.workflow_id for workflow in workflow_harness_catalog()}

    assert "visual-qa" in workflow_ids
    assert "ultraresearch" in workflow_ids

    missing_visual_surface = run_workflow_harness_smoke(
        "visual-qa",
        {"request_id": "visual-missing", "evidence": {}},
        tmp_path / "missing",
    )

    assert missing_visual_surface.status == "blocked"
    assert missing_visual_surface.blockers == ("visual_qa_surface_required",)
    assert missing_visual_surface.missing_evidence == ("surface_ref", "screenshot_ref", "oracle_verdict")

    visual_ready = run_workflow_harness_smoke(
        "visual-qa",
        {
            "request_id": "visual-ready",
            "goal_id": "goal-visual",
            "evidence": {
                "surface_ref": "tui://workflow-panel",
                "screenshot_ref": "screenshots/workflow-panel.txt",
                "oracle_verdict": {"status": "passed", "issues": []},
            },
        },
        tmp_path / "ready",
    )
    verdict_path = tmp_path / "ready" / "visual-qa" / "verdict.json"

    assert visual_ready.status == "ready"
    assert visual_ready.artifact_refs == ("visual-qa/surface-capture.json", "visual-qa/verdict.json")
    assert verdict_path.is_file()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["artifact_kind"] == "visual_qa_verdict"
    assert verdict["oracle_verdict"]["status"] == "passed"

    research_ready = run_workflow_harness_smoke(
        "ultraresearch",
        {
            "request_id": "research-ready",
            "goal_id": "goal-research",
            "evidence": {
                "research_question": "What public evidence supports ASA workflow parity?",
                "source_journal": "journals/workflow-research.jsonl",
                "insane_search_trace": {"skill_id": "insane_search", "grid_exhausted": False},
            },
        },
        tmp_path / "research",
    )
    acquisition_plan_path = tmp_path / "research" / "ultraresearch" / "acquisition-plan.json"
    journal_path = tmp_path / "research" / "ultraresearch" / "research-journal.jsonl"

    assert research_ready.status == "ready"
    assert research_ready.artifact_refs == (
        "ultraresearch/acquisition-plan.json",
        "ultraresearch/research-journal.jsonl",
    )
    assert acquisition_plan_path.is_file()
    assert journal_path.is_file()
    acquisition_plan = json.loads(acquisition_plan_path.read_text(encoding="utf-8"))
    assert acquisition_plan["artifact_kind"] == "ultraresearch_acquisition_plan"
    assert acquisition_plan["insane_search"]["surface"] == "skill"
    assert acquisition_plan["insane_search"]["skill_id"] == "insane_search"
    assert acquisition_plan["insane_search"]["public_only"] is True


def test_public_workflow_and_insane_search_surfaces_are_model_and_tui_visible(tmp_path: Path) -> None:
    from sim_agent.agent_harness.tools import (
        RuntimeToolCall,
        RuntimeToolError,
        default_tool_registry,
        execute_runtime_tool,
        tool_registry_for_agent,
    )
    from sim_agent.cli.tui_catalog import all_commands
    from sim_agent.cli.tui_workflow import WORKFLOW_ALIASES
    from sim_agent.cli.tui_state import initial_state

    registry = default_tool_registry()
    tools = {tool.name: tool for tool in registry.tools}
    workflow_ids = tools["workflow_start"].parameters["properties"]["workflow_id"]["enum"]
    skill_ids = tools["skill_invoke"].parameters["properties"]["skill_id"]["enum"]
    command_names = {command.name for command in all_commands()}

    assert "visual-qa" in workflow_ids
    assert "ultraresearch" in workflow_ids
    assert "insane_search" in skill_ids
    assert "/visual-qa" in command_names
    assert "/ultraresearch" in command_names
    assert "visual-qa" in WORKFLOW_ALIASES
    assert "ultraresearch" in WORKFLOW_ALIASES

    state = initial_state(tmp_path)
    invocation = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="skill_invoke",
            arguments={
                "skill_id": "insane_search",
                "payload": {"request_id": "insane-skill", "query": "ASA workflow parity public references"},
            },
            run_id="insane-search-run",
            session_id=state.session_id,
            caller_agent_id="research_agent",
        ),
        registry,
        state.session_dir,
    )
    adapter_output = invocation.output["result"]["adapter_output"]

    assert invocation.status == "ready"
    assert invocation.output["skill_id"] == "insane_search"
    assert adapter_output["surface"] == "skill"
    assert adapter_output["public_only"] is True
    assert adapter_output["ssrf_safe"] is True

    for agent_id in ("orchestrator", "md_agent", "ml_agent", "feature_scale_agent", "research_agent", "qa_agent"):
        registry = tool_registry_for_agent(agent_id)
        agent_tools = {tool.name: tool for tool in registry.tools}
        assert "workflow_start" in agent_tools
        assert "workflow_gate_response" in agent_tools
        assert "visual-qa" in agent_tools["workflow_start"].parameters["properties"]["workflow_id"]["enum"]
        assert "ultraresearch" in agent_tools["workflow_start"].parameters["properties"]["workflow_id"]["enum"]

    with pytest.raises(RuntimeToolError, match="unknown_agent_id"):
        tool_registry_for_agent("planner")
