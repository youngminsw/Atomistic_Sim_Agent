from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_gate_payload_none_is_blocker_not_ready(tmp_path: Path) -> None:
    # Given: RALPlan has nominal PRD/test-spec evidence but no persisted approval gate payload.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("ralplan", _ralplan_payload(), tmp_path)

    # Then: readiness fails closed instead of treating execution success as approval readiness.
    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("workflow_gate_payload_missing",)
    assert result.gate is None


def test_complete_gate_and_artifacts_can_be_ready(tmp_path: Path) -> None:
    # Given: RALPlan has evidence, generated artifacts, and an accepted approval gate.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    payload = _ralplan_payload(gate=_approval_gate())
    first = run_workflow_harness_smoke("ralplan", payload, tmp_path)
    accepted = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "ralplan",
            "gate_id": "approval",
            "responder_agent_id": "orchestrator",
            "value": {"decision": "approve"},
            "idempotency_key": "approve-001",
        },
    )

    # When: the harness is started again with the same gate and artifacts.
    second = run_workflow_harness_smoke("ralplan", payload, tmp_path)

    # Then: readiness is allowed only after the accepted gate is durable.
    assert first.status == "blocked"
    assert first.gate_status == "awaiting_response"
    assert accepted.status == "accepted"
    assert second.status == "ready"
    assert second.gate_status == "passed"
    assert second.gate is not None
    assert second.gate["status"] == "accepted"
    assert "ralplan/consensus.json" in second.artifact_refs


def test_weak_artifacts_and_failed_verdicts_block_readiness(tmp_path: Path) -> None:
    # Given: artifact evidence covers escape paths, missing files, visual verdict failure, and hash sabotage.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    outside = tmp_path.parent / "escaped-prd.md"
    outside.write_text("# outside\n", encoding="utf-8")

    escaped = run_workflow_harness_smoke(
        "ralplan",
        _ralplan_payload(
            gate=_approval_gate(),
            extra={
                "validate_artifact_paths": True,
                "artifact_root": str(tmp_path / "artifacts"),
                "evidence": {"prd_path": str(outside), "test_spec_path": "test-spec.md"},
            },
        ),
        tmp_path / "escaped",
    )
    missing = run_workflow_harness_smoke(
        "ralplan",
        _ralplan_payload(
            gate=_approval_gate(),
            extra={
                "validate_artifact_paths": True,
                "artifact_root": str(tmp_path / "artifacts"),
                "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
            },
        ),
        tmp_path / "missing",
    )
    visual = run_workflow_harness_smoke("visual-qa", _visual_payload(tmp_path, passed=False), tmp_path / "visual")
    verifier = subprocess.run(
        [
            sys.executable,
            "scripts/check_workflow_gap_evidence.py",
            "--manifest",
            "tests/fixtures/workflow_gap/final-manifest.json",
            "--evidence-dir",
            "tests/fixtures/workflow_gap",
            "--parity",
            "tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.md",
        ],
        cwd=SOURCE_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )

    # Then: every weak-evidence route has a typed blocker and no readiness false-green.
    assert escaped.status == "blocked"
    assert escaped.blockers == ("workflow_artifact_path_untrusted",)
    assert missing.status == "blocked"
    assert missing.blockers == ("ralplan_artifact_missing",)
    assert visual.status == "blocked"
    assert visual.blockers == ("visual_qa_verdict_failed",)
    assert verifier.returncode == 0, verifier.stdout + verifier.stderr
    assert "workflow_gap_evidence_status=passed" in verifier.stdout


def _approval_gate() -> JsonMap:
    return {
        "gate_id": "approval",
        "gate_kind": "response_schema",
        "response_schema": {
            "type": "object",
            "required": ["decision"],
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "request-changes", "reject"]},
                "comments": {"type": "string"},
            },
        },
    }


def _ralplan_payload(*, gate: JsonMap | None = None, extra: JsonMap | None = None) -> JsonMap:
    payload: JsonMap = {
        "request_id": "ralplan-readiness",
        "user_goal": "Harden workflow readiness",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-readiness",
        "evidence": {"prd_path": "prd.md", "test_spec_path": "test-spec.md"},
    }
    if gate is not None:
        payload["gate"] = gate
    if extra is not None:
        payload.update(extra)
    return payload


def _visual_payload(root: Path, *, passed: bool) -> JsonMap:
    screenshot = root / "visual" / "screenshots" / "workflow.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    screenshot.write_text("visual capture bytes\n", encoding="utf-8")
    return {
        "request_id": "visual-readiness",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-visual-readiness",
        "evidence": {
            "surface_ref": "app://asa/workflow",
            "screenshot_ref": "screenshots/workflow.png",
            "oracle_verdict": {"passed": passed, "summary": "readiness check"},
        },
    }
