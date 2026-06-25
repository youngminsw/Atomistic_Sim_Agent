from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from sim_agent.cli.tui_semantic import filter_semantic_tty_output
from sim_agent.cli.tui_tools import handle_tools


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_tui_control_room_smoke_cli_writes_parity_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            "--tui-control-room-smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=SOURCE_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=70,
    )

    matrix_path = output_dir / "task-11-tui-command-parity-matrix.json"
    transcript_path = output_dir / "task-11-tui-transcript.txt"
    final_transcript_path = output_dir / "final-f3-tui-transcript.txt"

    assert result.returncode == 0, result.stdout + result.stderr
    assert "tui_control_room_smoke_status=succeeded" in result.stdout
    assert matrix_path.is_file()
    assert transcript_path.is_file()
    assert final_transcript_path.is_file()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["status"] == "succeeded"
    assert matrix["blockers"] == []
    assert all(matrix["checks"].values())
    assert matrix["checks"]["forced_tool_call_activity"] is True
    transcript = transcript_path.read_text(encoding="utf-8")
    assert "ASA Activity Rail" in transcript
    assert "Tool Call" in transcript
    assert "artifact_write" in transcript
    assert "event=tool_start status=running detail=artifact_write" in transcript
    assert "event=tool_end status=ok detail=artifact_write" in transcript
    assert "Runtime Tool Catalog" in transcript
    assert "Workflow Catalog" in transcript
    assert "ASA Chat Deck" in transcript


def test_tui_tools_human_rows_survive_tty_semantic_filter() -> None:
    output = _TtyStringIO()
    filtered = filter_semantic_tty_output(output)

    handle_tools(filtered)

    rendered = output.getvalue()
    assert "Runtime Tool Catalog" in rendered
    assert "bash_process" in rendered
    assert "mcp_list_tools" in rendered
    assert "tool=bash_process" not in rendered
    assert "approval_required=" not in rendered


class _TtyStringIO(StringIO):
    @property
    def encoding(self) -> str:
        return "utf-8"

    def isatty(self) -> bool:
        return True
