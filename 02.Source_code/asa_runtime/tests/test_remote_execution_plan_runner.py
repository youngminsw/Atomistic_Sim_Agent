from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import run_remote_execution_plan


def test_remote_execution_plan_runner_completes_ordered_commands(tmp_path: Path) -> None:
    plan_path = tmp_path / "remote_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "ssh_target": "local-test",
                "ssh_port": 22,
                "all_commands": [
                    "echo first",
                    "echo second",
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_remote_execution_plan(plan_path, timeout_s=5)

    assert result.ok is True
    assert result.payload["plan_status"] == "remote_plan_completed"
    assert result.payload["completed_command_count"] == 2
    assert "second" in result.payload["stdout_tail"]


def test_remote_execution_plan_runner_records_first_failure(tmp_path: Path) -> None:
    plan_path = tmp_path / "remote_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "ssh_target": "local-test",
                "ssh_port": 22,
                "all_commands": [
                    "echo before",
                    "echo failed >&2; exit 7",
                    "echo after",
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_remote_execution_plan(plan_path, timeout_s=5)

    assert result.ok is False
    assert result.payload["plan_status"] == "remote_plan_failed"
    assert result.payload["returncode"] == 7
    assert result.payload["completed_command_count"] == 1
    assert result.payload["failed_command"] == "echo failed >&2; exit 7"
    assert "remote_plan_command_failed" in result.payload["blockers"]
