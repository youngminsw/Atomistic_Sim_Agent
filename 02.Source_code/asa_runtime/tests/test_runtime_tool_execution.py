from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_runtime_tools_execute_bash_artifact_and_graphdb_dry_run(tmp_path: Path) -> None:
    # Given: the default executable runtime tool registry.
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    registry = default_tool_registry()

    # When: the runtime executes safe Bash, artifact, and GraphDB dry-run tools.
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
    assert artifact.status == "succeeded"
    assert artifact.output["relative_path"] == "notes/tool.txt"
    assert (tmp_path / "artifacts" / "notes" / "tool.txt").read_text(encoding="utf-8") == "runtime evidence"
    assert (tmp_path / artifact.artifact_ref).is_file()
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
