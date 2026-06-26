from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_workflow_e2e_smoke_cli_drives_all_workflow_surfaces(tmp_path: Path) -> None:
    # Given: an output directory for a deterministic full-workflow e2e smoke.
    output_dir = tmp_path / "final-e2e"

    # When: the user-visible workflow e2e CLI is driven through the real command surface.
    result = _run_cli("--workflow-e2e-smoke", output_dir, "full-workflow-loop")

    # Then: stdout, ledger, transcript, and artifacts prove all workflow surfaces ran with one run id.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_e2e_smoke_status=succeeded" in result.stdout
    payload = json.loads((output_dir / "workflow-e2e.json").read_text(encoding="utf-8"))
    transcript = (output_dir / "workflow-transcript.txt").read_text(encoding="utf-8")
    run_id = payload["run_id"]

    assert payload["status"] == "succeeded"
    assert payload["workflow_ids"] == ["/deep-interview", "/ralplan", "/ultragoal", "/visual-qa", "/ultraresearch"]
    assert payload["skill_ids"] == ["insane-search"]
    assert all(row["status"] == "ready" for row in payload["workflow_results"])
    assert all(row["gate_status"] == "passed" for row in payload["workflow_results"])
    assert all(row["blockers"] == [] for row in payload["workflow_results"])
    assert payload["bounded_subagent_denials"] == {
        "/deep-interview": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ralplan": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ultragoal": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/visual-qa": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "/ultraresearch": "persistent_workflow_surface_unavailable_for_bounded_subagent",
        "insane-search": "persistent_skill_surface_unavailable_for_bounded_subagent",
    }
    assert f"run_id={run_id}" in transcript
    for command in payload["workflow_ids"]:
        assert command in transcript
    assert "skill=insane-search" in transcript
    for artifact in payload["artifacts"]:
        artifact_text = (output_dir / artifact).read_text(encoding="utf-8")
        assert run_id in artifact_text


def test_workflow_live_llm_e2e_cli_blocks_without_explicit_live_flag(tmp_path: Path) -> None:
    # Given: no ASA_LIVE_LLM_E2E opt-in in the environment.
    output_dir = tmp_path / "final-live-llm"

    # When: the live workflow e2e command is invoked.
    result = _run_cli("--workflow-live-llm-e2e", output_dir, "all-workflows")

    # Then: the command records typed provider-unavailable evidence and does not claim success.
    assert result.returncode == 1
    assert "workflow_live_llm_e2e_status=blocked" in result.stdout
    assert "workflow_live_llm_e2e_blocker=live_llm_provider_unavailable" in result.stdout
    payload = json.loads((output_dir / "workflow-live-llm.json").read_text(encoding="utf-8"))
    provider_events = (output_dir / "provider-events.jsonl").read_text(encoding="utf-8")

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["live_llm_provider_unavailable"]
    assert payload["live_llm"][0]["status"] == "blocked"
    assert payload["live_llm"][0]["blocker"] == "live_llm_provider_unavailable"
    assert "live_llm_provider_unavailable" in provider_events


def _run_cli(flag: str, output_dir: Path, scenario: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("ASA_LIVE_LLM_E2E", None)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            flag,
            "--scenario",
            scenario,
            "--output-dir",
            str(output_dir),
        ],
        cwd=SOURCE_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
