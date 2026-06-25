from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_asa_interactive_tools_surface_lists_executable_tool_status(tmp_path: Path) -> None:
    # Given: an interactive ASA session.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")

    # When: the user opens the tool catalog from the slash-first TUI.
    result = subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="/tools\n/help\n/exit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: executable runtime tool safety status is visible without long CLI flags.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tool_catalog=true" in result.stdout
    assert "tool=bash_process" in result.stdout
    assert "tool=shell_command" in result.stdout
    assert "family=process" in result.stdout
    assert "safety=workspace_write" in result.stdout
    assert "policy_id=safe-smoke-process-v1" in result.stdout
    assert "policy=local_smoke:exact_argv_allowlist:3" in result.stdout
    assert "tool=artifact_write" in result.stdout
    assert "tool=file_read" in result.stdout
    assert "tool=file_write" in result.stdout
    assert "tool=file_search" in result.stdout
    assert "tool=file_edit" in result.stdout
    assert "tool=mcp_list_tools" in result.stdout
    assert "family=mcp" in result.stdout
    assert "tool=custom_tool_register" in result.stdout
    assert "family=custom_tool" in result.stdout
    assert "tool=graphdb_dry_run" in result.stdout
    assert "approval_required=false" in result.stdout
    assert "/tools" in result.stdout
