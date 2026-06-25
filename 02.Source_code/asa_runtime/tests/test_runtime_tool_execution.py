from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sim_agent.schemas._parse import as_mapping, as_sequence, as_str


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_runtime_tools_execute_bash_files_artifact_mcp_custom_and_graphdb_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the default executable runtime tool registry.
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text(
        """{
  "mcpServers": {
    "paper-memory": {
      "transport": "stdio",
      "command": "python3",
      "args": ["fake_mcp.py"],
      "tools": [
        {
          "name": "neo4j_search",
          "description": "Fake source-backed graph search.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
      ]
    }
  }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    registry = default_tool_registry()

    # When: the runtime executes safe Bash, file, artifact, MCP/custom descriptor, and GraphDB dry-run tools.
    bash = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="bash_process",
            arguments={"argv": ["python3", "-c", "print('asa-tool-ok')"]},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    shell = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="shell_command",
            arguments={"argv": ["python3", "--version"]},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    file_write = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_write",
            arguments={"relative_path": "notes/tool.txt", "content": "alpha\nneedle\n"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    file_read = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_read",
            arguments={"relative_path": "notes/tool.txt"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    file_search = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_search",
            arguments={"query": "needle"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    file_edit = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="file_edit",
            arguments={
                "relative_path": "notes/tool.txt",
                "search": "needle",
                "replace": "replacement",
                "expected_replacements": 1,
            },
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    artifact = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="artifact_write",
            arguments={"relative_path": "notes/tool.txt", "content": "runtime evidence"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    mcp = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="mcp_list_tools",
            arguments={"server_name": "paper-memory"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    custom = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="custom_tool_register",
            arguments={
                "name": "local_probe",
                "description": "Local descriptor-only custom tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )
    graphdb = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="graphdb_dry_run",
            arguments={"database_name": "atomistic_sim_agent_knowledge"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        registry,
        tmp_path,
    )

    # Then: each call returns structured success evidence and writes an artifact ledger.
    assert bash.status == "succeeded"
    assert bash.output["stdout"] == "asa-tool-ok\n"
    assert (tmp_path / bash.artifact_ref).is_file()
    assert shell.status == "succeeded"
    assert (tmp_path / shell.artifact_ref).is_file()
    assert file_write.status == "succeeded"
    assert file_read.status == "succeeded"
    assert file_read.output["content"] == "alpha\nneedle\n"
    assert file_search.status == "succeeded"
    assert file_search.output["match_count"] == 1
    assert file_edit.status == "succeeded"
    assert (tmp_path / "workspace" / "notes" / "tool.txt").read_text(encoding="utf-8") == "alpha\nreplacement\n"
    assert (tmp_path / file_edit.artifact_ref).is_file()
    assert artifact.status == "succeeded"
    assert artifact.output["relative_path"] == "notes/tool.txt"
    assert (tmp_path / "artifacts" / "notes" / "tool.txt").read_text(encoding="utf-8") == "runtime evidence"
    assert (tmp_path / artifact.artifact_ref).is_file()
    assert mcp.status == "succeeded"
    assert mcp.output["server_name"] == "paper-memory"
    assert mcp.output["transport"] == "stdio"
    mcp_tool = as_mapping(as_sequence(mcp.output["tools"], "tools")[0], "tool")
    assert as_str(mcp_tool["name"], "name") == "neo4j_search"
    assert custom.status == "succeeded"
    assert (tmp_path / "custom_tools" / "local_probe.json").is_file()
    assert graphdb.status == "succeeded"
    assert graphdb.output["neo4j_write_enabled"] is False
    assert graphdb.output["database_name"] == "atomistic_sim_agent_knowledge"
    assert (tmp_path / graphdb.artifact_ref).is_file()


def test_runtime_bash_tool_blocks_destructive_commands(tmp_path: Path) -> None:
    # Given: a destructive shell command request.
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    # When: the request is executed through the runtime tool layer.
    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="bash_process",
            arguments={"argv": ["rm", "-rf", "/tmp/asa-should-not-run"]},
            run_id="tool-run",
            session_id="tool-session",
        ),
        default_tool_registry(),
        tmp_path,
    )

    # Then: the runtime blocks it before subprocess execution.
    assert result.status == "blocked"
    assert result.blocker == "unsafe_command"
    assert (tmp_path / result.artifact_ref).is_file()


def test_runtime_bash_tool_blocks_shell_bypass_and_path_traversal(tmp_path: Path) -> None:
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    shell = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="bash_process",
            arguments={"argv": ["sh", "-c", "touch escaped"]},
            run_id="tool-run",
            session_id="tool-session",
        ),
        default_tool_registry(),
        tmp_path,
    )
    traversal = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="bash_process",
            arguments={"argv": ["python3", "-c", "print('asa-tool-ok')"]},
            run_id="../../../escape_run",
            session_id="tool-session",
        ),
        default_tool_registry(),
        tmp_path,
    )

    assert shell.status == "blocked"
    assert shell.blocker == "unsafe_command"
    assert not (tmp_path / "escaped").exists()
    assert traversal.status == "blocked"
    assert traversal.blocker == "invalid_run_id"
    assert traversal.artifact_ref == "tool_ledgers/invalid-run-id/bash_process.json"
    assert (tmp_path / traversal.artifact_ref).is_file()
    assert not (tmp_path.parent / "escape_run").exists()
