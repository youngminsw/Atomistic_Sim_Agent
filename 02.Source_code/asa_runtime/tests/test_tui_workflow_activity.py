from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_tui_workflow_response_surface_shows_gate_owner_goal_and_ledger(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    result = _run_tui(
        tmp_path,
        (
            "/workflow deep-interview --owner-agent orchestrator --target-agent orchestrator --goal-id goal-di "
            "--deep-question-id clarify --deep-ambiguity 0.2 --deep-options clear,blocked "
            f"--output-dir {workflow_dir}\n"
            "/workflow-response question-clarify '{\"selected\":[\"clear\"]}' --workflow-id deep-interview "
            f"--responder-agent orchestrator --output-dir {workflow_dir}\n"
            "/ralplan --evidence-key prd_path,test_spec_path --owner-agent orchestrator "
            "--target-agent qa_agent --goal-id goal-ral --gate-id approval "
            f"--allowed-values approve,revise --output-dir {workflow_dir}\n"
            "/workflow-response approval approve --workflow-id ralplan "
            f"--responder-agent qa_agent --output-dir {workflow_dir}\n"
            "/ultragoal --evidence-key codex_goal_snapshot --owner-agent orchestrator "
            f"--target-agent orchestrator --goal-id goal-ultra --output-dir {workflow_dir}\n"
            "/help\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow=deep-interview" in result.stdout
    assert "workflow_status=blocked" in result.stdout
    assert "workflow_gate_status=awaiting_response" in result.stdout
    assert "workflow_gate_id=question-clarify" in result.stdout
    assert "workflow_owner_agent_id=orchestrator" in result.stdout
    assert "workflow_target_agent_id=orchestrator" in result.stdout
    assert "workflow_goal_id=goal-di" in result.stdout
    assert "workflow_loop_state=blocked" in result.stdout
    assert "workflow_response=true" in result.stdout
    assert "workflow_response_status=accepted" in result.stdout
    assert "workflow=ralplan" in result.stdout
    assert "workflow_gate_id=approval" in result.stdout
    assert "workflow_target_agent_id=qa_agent" in result.stdout
    assert "workflow=ultragoal" in result.stdout
    assert "workflow_goal_id=goal-ultra" in result.stdout
    assert "workflow_artifact_refs=ultragoal/brief.md,ultragoal/goals.json,ultragoal/ledger.jsonl" in result.stdout
    assert "/workflow-response <gate-id> <value>" in result.stdout

    deep_gate = workflow_dir / "deep-interview" / "gates" / "question-clarify.json"
    ralplan_gate = workflow_dir / "ralplan" / "gates" / "approval.json"
    assert deep_gate.is_file()
    assert ralplan_gate.is_file()
    assert (workflow_dir / "deep-interview" / "handoff.md").is_file()
    assert (workflow_dir / "ralplan" / "prd.md").is_file()
    assert (workflow_dir / "ralplan" / "test-spec.md").is_file()
    assert (workflow_dir / "ultragoal" / "brief.md").is_file()
    assert (workflow_dir / "ultragoal" / "goals.json").is_file()
    assert (workflow_dir / "ultragoal" / "ledger.jsonl").is_file()
    assert json.loads(deep_gate.read_text(encoding="utf-8"))["status"] == "accepted"
    assert json.loads(ralplan_gate.read_text(encoding="utf-8"))["target_agent_id"] == "qa_agent"


def test_tui_workflow_surface_shows_domain_self_and_peer_authority_blocker(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    result = _run_tui(
        tmp_path,
        (
            "/workflow ultrawork --evidence-key lane_outputs --actor-agent md_agent "
            f"--owner-agent md_agent --goal-id goal-md --output-dir {workflow_dir}\n"
            "/workflow ultrawork --evidence-key lane_outputs --actor-agent md_agent "
            f"--owner-agent md_agent --target-agent qa_agent --goal-id goal-peer --output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow=ultrawork" in result.stdout
    assert "workflow_actor_agent_id=md_agent" in result.stdout
    assert "workflow_owner_agent_id=md_agent" in result.stdout
    assert "workflow_target_agent_id=md_agent" in result.stdout
    assert "workflow_goal_id=goal-md" in result.stdout
    assert "workflow_status=ready" in result.stdout
    assert "workflow_target_agent_id=qa_agent" in result.stdout
    assert "workflow_goal_id=goal-peer" in result.stdout
    assert "workflow_status=blocked" in result.stdout
    assert "workflow_blocker=workflow_authority_peer_denied" in result.stdout


def test_tui_workflow_response_preserves_cli_enum_values_as_strings(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    result = _run_tui(
        tmp_path,
        (
            "/workflow ralplan --evidence-key prd_path,test_spec_path "
            "--gate-id numeric --allowed-values 1,2 "
            f"--output-dir {workflow_dir}\n"
            f"/workflow-response numeric 1 --workflow-id ralplan --output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_response_status=accepted" in result.stdout
    gate_payload = json.loads((workflow_dir / "ralplan" / "gates" / "numeric.json").read_text(encoding="utf-8"))
    assert gate_payload["response_value"] == "1"


def test_tui_workflow_response_parses_response_schema_json_scalars(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    result = _run_tui(
        tmp_path,
        (
            "/workflow ralplan --evidence-key prd_path,test_spec_path "
            "--gate-id confirmed --gate-kind response_schema --response-schema '{\"type\":\"boolean\"}' "
            f"--output-dir {workflow_dir}\n"
            f"/workflow-response confirmed true --workflow-id ralplan --output-dir {workflow_dir}\n"
            "/workflow ralplan --evidence-key prd_path,test_spec_path "
            "--gate-id score --gate-kind response_schema --response-schema '{\"type\":\"number\"}' "
            f"--output-dir {workflow_dir}\n"
            f"/workflow-response score 1 --workflow-id ralplan --output-dir {workflow_dir}\n"
            "/workflow ralplan --evidence-key prd_path,test_spec_path "
            "--gate-id optional --gate-kind response_schema --response-schema '{\"type\":\"null\"}' "
            f"--output-dir {workflow_dir}\n"
            f"/workflow-response optional null --workflow-id ralplan --output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("workflow_response_status=accepted") == 3
    gate_payload = json.loads((workflow_dir / "ralplan" / "gates" / "confirmed.json").read_text(encoding="utf-8"))
    assert gate_payload["response_value"] is True
    gate_payload = json.loads((workflow_dir / "ralplan" / "gates" / "score.json").read_text(encoding="utf-8"))
    assert gate_payload["response_value"] == 1
    gate_payload = json.loads((workflow_dir / "ralplan" / "gates" / "optional.json").read_text(encoding="utf-8"))
    assert gate_payload["response_value"] is None


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
