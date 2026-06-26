from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_visual_qa_materializes_machine_checked_surface_and_verdict(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("visual-qa", _payload(tmp_path), tmp_path)

    assert result.status == "ready"
    assert result.gate_status == "passed"
    assert result.artifact_refs == (
        "visual-qa/surface-capture.json",
        "visual-qa/verdict.json",
        "visual-qa/evidence-ledger.jsonl",
    )
    surface = _read_json(tmp_path / "visual-qa" / "surface-capture.json")
    verdict = _read_json(tmp_path / "visual-qa" / "verdict.json")
    ledger = _jsonl(tmp_path / "visual-qa" / "evidence-ledger.jsonl")
    assert surface["schema_version"] == "visual_qa_surface_v1"
    assert surface["surface_ref"] == "app://asa/workflow"
    assert surface["screenshot_ref"] == "screenshots/workflow.png"
    assert surface["screenshot_size_bytes"] > 0
    assert isinstance(surface["screenshot_sha256"], str)
    assert verdict["schema_version"] == "visual_qa_verdict_v1"
    assert verdict["machine_checked"] is True
    assert verdict["passed"] is True
    assert ledger[0]["passed"] is True


def test_visual_qa_blocks_malformed_oracle_verdict(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(tmp_path)
    payload["evidence"]["oracle_verdict"] = {"summary": "missing pass boolean"}

    result = run_workflow_harness_smoke("visual-qa", payload, tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("visual_qa_verdict_invalid",)
    assert not (tmp_path / "visual-qa" / "verdict.json").exists()


def test_visual_qa_blocks_failed_oracle_verdict(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(tmp_path)
    payload["evidence"]["oracle_verdict"] = {"passed": False, "summary": "overlapping controls"}

    result = run_workflow_harness_smoke("visual-qa", payload, tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("visual_qa_verdict_failed",)
    assert not (tmp_path / "visual-qa" / "verdict.json").exists()


def test_visual_qa_blocks_missing_screenshot_file(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(tmp_path)
    (tmp_path / "screenshots" / "workflow.png").unlink()

    result = run_workflow_harness_smoke("visual-qa", payload, tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("visual_qa_screenshot_missing",)
    assert not (tmp_path / "visual-qa" / "verdict.json").exists()


def test_visual_qa_blocks_untrusted_screenshot_path(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    outside = tmp_path.parent / "outside-visual-capture.txt"
    outside.write_text("outside capture\n", encoding="utf-8")
    payload = _payload(tmp_path)
    payload["evidence"]["screenshot_ref"] = str(outside)

    result = run_workflow_harness_smoke("visual-qa", payload, tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("visual_qa_screenshot_untrusted",)
    assert not (tmp_path / "visual-qa" / "verdict.json").exists()


def test_visual_qa_artifacts_are_idempotent_and_block_tampered_surface(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(tmp_path)
    first = run_workflow_harness_smoke("visual-qa", payload, tmp_path)
    surface = tmp_path / "visual-qa" / "surface-capture.json"
    original_ledger = (tmp_path / "visual-qa" / "evidence-ledger.jsonl").read_text(encoding="utf-8")
    surface.write_text('{"tampered": true}\n', encoding="utf-8")

    second = run_workflow_harness_smoke("visual-qa", payload, tmp_path)

    assert first.status == "ready"
    assert second.status == "blocked"
    assert second.gate_status == "blocked"
    assert second.blockers == ("visual_qa_artifact_conflict",)
    assert surface.read_text(encoding="utf-8") == '{"tampered": true}\n'
    assert (tmp_path / "visual-qa" / "evidence-ledger.jsonl").read_text(encoding="utf-8") == original_ledger


def test_tui_visual_qa_accepts_structured_evidence_options(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    _write_capture(workflow_dir / "screenshots" / "workflow.png")
    result = _run_tui(
        tmp_path,
        (
            "/workflow visual-qa --owner-agent orchestrator --target-agent orchestrator --goal-id goal-visual "
            "--surface-ref app://asa/workflow --screenshot-ref screenshots/workflow.png "
            "--oracle-verdict '{\"passed\":true,\"summary\":\"clean\"}' "
            f"--output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow=visual-qa" in result.stdout
    assert "workflow_status=ready" in result.stdout
    assert "workflow_artifact_refs=visual-qa/surface-capture.json,visual-qa/verdict.json,visual-qa/evidence-ledger.jsonl" in result.stdout
    verdict = _read_json(workflow_dir / "visual-qa" / "verdict.json")
    assert verdict["passed"] is True
    assert verdict["oracle_verdict"]["summary"] == "clean"


def _payload(root: Path) -> JsonMap:
    _write_capture(root / "screenshots" / "workflow.png")
    return {
        "request_id": "visual-rich",
        "user_goal": "Verify rendered workflow UI",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-visual-rich",
        "evidence": {
            "surface_ref": "app://asa/workflow",
            "screenshot_ref": "screenshots/workflow.png",
            "oracle_verdict": {"passed": True, "summary": "clean", "checks": ["layout", "state"]},
        },
    }


def _write_capture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("visual capture bytes\n", encoding="utf-8")


def _read_json(path: Path) -> JsonMap:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _jsonl(path: Path) -> list[JsonMap]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_tui(tmp_path: Path, input_text: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ASA_SESSION_DIR"] = str(tmp_path / "session")
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=SOURCE_ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
