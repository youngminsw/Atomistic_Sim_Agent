from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_skill_workflow_smoke_cli_writes_skill_and_workflow_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "g006-evidence"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            "--skill-workflow-smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=SOURCE_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=45,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    skill_matrix = json.loads((output_dir / "task-8-skill-parity-matrix.json").read_text(encoding="utf-8"))
    workflow_matrix = json.loads((output_dir / "task-10-workflow-gate-parity-matrix.json").read_text(encoding="utf-8"))
    transcript = (output_dir / "task-8-skills.txt").read_text(encoding="utf-8")

    assert "skill_workflow_smoke_status=succeeded" in result.stdout
    assert skill_matrix["status"] == "succeeded"
    assert workflow_matrix["status"] == "succeeded"
    assert skill_matrix["skill_sources"] == {
        ".asa/skills": True,
        ".codex/skills": True,
        ".claude/skills": True,
    }
    assert {"/asa-probe", "/codex-probe", "/claude-probe"}.issubset(skill_matrix["palette_commands_present"])
    assert skill_matrix["slash_invocation_count"] == 3
    assert skill_matrix["prompt_context_record_count"] == 3
    assert workflow_matrix["missing_case"]["gate_status"] == "blocked"
    assert workflow_matrix["passed_case"]["gate_status"] == "passed"
    assert "workflow_missing_evidence=prd_path,test_spec_path" in transcript
    assert "workflow_evidence_keys=prd_path,test_spec_path" in transcript
