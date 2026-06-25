from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_compaction_smoke_cli_writes_compaction_resume_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"

    result = _run_compaction_smoke(output_dir)

    matrix_path = output_dir / "task-9-compaction-parity-matrix.json"
    transcript_path = output_dir / "task-9-compaction.txt"
    e2e_path = output_dir / "final-f3-e2e.json"
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compaction_smoke=true" in result.stdout
    assert matrix_path.is_file()
    assert transcript_path.is_file()
    assert e2e_path.is_file()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    transcript = transcript_path.read_text(encoding="utf-8")
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    assert matrix["status"] == "succeeded"
    assert matrix["blockers"] == []
    assert matrix["checks"]["manual_compact_replayed"] is True
    assert matrix["checks"]["auto_threshold_compacted"] is True
    assert matrix["checks"]["poison_blocked"] is True
    assert matrix["checks"]["stale_cursor_blocked"] is True
    assert matrix["checks"]["orphan_tool_result_blocked"] is True
    assert matrix["checks"]["prompt_manifest_has_compact_summary_layer"] is True
    assert matrix["checks"]["prompt_manifest_has_validated_summary"] is True
    assert matrix["poison"]["blocker"] == "compact_summary_poisoned"
    assert matrix["stale_cursor"]["blocker"] == "stale_compact_cursor"
    assert matrix["orphan_tool_result"]["blocker"] == "orphan_tool_result"
    assert matrix["resume"]["opened_as"] == "resumed"
    assert matrix["resume"]["turn"]["status"] == "succeeded"
    assert "compact_replay_status=replayed" in transcript
    assert "prompt_manifest_layer_kinds=" in transcript
    assert e2e["status"] == "succeeded"
    assert e2e["provider_prompt_manifest"]["has_compact_summary_layer"] is True


def _run_compaction_smoke(output_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            "--compaction-smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )
