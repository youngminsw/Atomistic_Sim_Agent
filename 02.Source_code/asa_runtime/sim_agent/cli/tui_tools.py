from __future__ import annotations

from typing import TextIO

from sim_agent.agent_harness import default_tool_registry


def handle_tools(output_stream: TextIO) -> None:
    registry = default_tool_registry()
    output_stream.write("Runtime Tool Catalog\n")
    output_stream.write("tool_catalog=true\n")
    for tool in registry.tools:
        if tool.executable:
            output_stream.write(
                f"tool={tool.name} family={tool.family} safety={tool.safety} "
                f"approval_required={str(tool.approval_required).lower()} executable=true "
                f"policy_id={tool.policy_id} policy={tool.policy_summary}\n"
            )
