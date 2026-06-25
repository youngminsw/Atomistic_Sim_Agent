from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_adversarial_e2e_smoke_cli_writes_blocker_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "sim_agent.cli.main", "--adversarial-e2e-smoke", "--output-dir", str(output_dir)],
        cwd=SOURCE_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "adversarial_e2e_smoke=true" in result.stdout
    payload = json.loads((output_dir / "ultraqa" / "adversarial-e2e.json").read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["blockers"] == []
    assert {case["expected_blocker"] for case in payload["provider_cases"]} >= {
        "no_model_tool_selected",
        "unknown_model_tool_selected",
        "unsafe_model_tool_selected",
        "malformed_model_tool_call",
    }
    assert {case["expected_blocker"] for case in payload["subagent_cases"]} >= {
        "duplicate_task_id",
        "unknown_preset",
        "subagent_depth_exceeded",
        "subagent_recursion_blocked",
        "too_many_active_subagents",
    }
    assert {case["expected_blocker"] for case in payload["compaction_cases"]} >= {
        "corrupt_ledger",
        "stale_compact_cursor",
        "compact_summary_poisoned",
        "orphan_tool_result",
    }
    assert payload["destructive_writes_ran"] == {"graphdb": False, "md": False, "remote": False}
    assert payload["secret_redaction"]["token_leaked"] is False
    assert "asa-adversarial-secret-token" not in json.dumps(payload)
