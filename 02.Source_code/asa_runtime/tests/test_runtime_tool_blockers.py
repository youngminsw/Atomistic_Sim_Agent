from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_runtime_file_custom_and_unknown_tools_write_blocker_receipts(tmp_path: Path) -> None:
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    registry = default_tool_registry()

    traversal = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_write",
            arguments={"relative_path": "../escape.txt", "content": "nope"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    missing_root = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_search",
            arguments={"query": "x", "root": "../outside"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    mismatch = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="custom_tool_register",
            arguments={
                "name": "bad_schema",
                "description": "bad",
                "parameters": {"type": "object", "properties": {}, "required": ["missing"]},
            },
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    malformed_mcp = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="mcp_list_tools",
            arguments={"server_name": {"not": "a-string"}},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    unknown = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="not_a_tool",
            arguments={},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    non_executable = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="validate_simulation_request",
            arguments={},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )

    assert traversal.status == "blocked"
    assert traversal.blocker == "unsafe_file_path"
    assert not (tmp_path.parent / "escape.txt").exists()
    assert missing_root.status == "blocked"
    assert missing_root.blocker == "unsafe_file_path"
    assert mismatch.status == "blocked"
    assert mismatch.blocker == "invalid_custom_tool_schema"
    assert malformed_mcp.status == "blocked"
    assert malformed_mcp.blocker == "invalid_arguments"
    assert (tmp_path / malformed_mcp.artifact_ref).is_file()
    assert unknown.status == "blocked"
    assert unknown.blocker == "unknown_tool"
    assert (tmp_path / unknown.artifact_ref).is_file()
    assert non_executable.status == "blocked"
    assert non_executable.blocker == "tool_not_executable"
    assert (tmp_path / non_executable.artifact_ref).is_file()
