from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class VerifierFixture:
    evidence_dir: Path
    manifest: Path
    parity: Path
    run_id: str


def test_workflow_gap_evidence_verifier_accepts_complete_manifest(tmp_path: Path) -> None:
    # Given: a manifest with parity evidence, e2e ledger/transcript, sabotage, and live provider proof.
    fixture = _write_valid_fixture(tmp_path)

    # When: the workflow evidence verifier runs through its CLI surface.
    result = _run_verifier(fixture)

    # Then: the manifest is accepted with a machine-readable success line.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_gap_evidence_status=passed" in result.stdout


def test_workflow_gap_evidence_verifier_rejects_false_green_inputs(tmp_path: Path) -> None:
    cases: dict[str, tuple[Callable[[VerifierFixture], None], str]] = {
        "missing_parity_evidence": (_remove_parity_evidence, "parity_evidence_missing"),
        "missing_negative_control": (_remove_negative_control, "parity_negative_control_missing"),
        "fake_stdout_no_ledger": (_remove_e2e_ledger, "e2e_ledger_missing"),
        "transcript_only_completion": (_remove_required_transcript_line, "e2e_transcript_missing_required"),
        "stale_artifact_run_id": (_write_stale_artifact, "artifact_run_id_mismatch"),
        "sabotage_not_detected": (_mark_sabotage_missed, "sabotage_case_not_detected"),
        "fake_live_provider": (_mark_live_provider_fake, "live_llm_fake_provider"),
        "provider_unavailable_not_success": (_mark_live_provider_unavailable, "live_llm_provider_unavailable"),
    }

    for case_name, (sabotage, expected_blocker) in cases.items():
        # Given: a valid manifest copy with one deliberate false-green sabotage.
        fixture = _write_valid_fixture(tmp_path / case_name)
        sabotage(fixture)

        # When: the verifier checks the sabotaged manifest.
        result = _run_verifier(fixture)

        # Then: the verifier fails for the typed blocker named by the sabotage.
        assert result.returncode == 1, case_name + result.stdout + result.stderr
        assert f"workflow_gap_evidence_blocker={expected_blocker}" in result.stdout


def _run_verifier(fixture: VerifierFixture) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_workflow_gap_evidence.py",
            "--manifest",
            str(fixture.manifest),
            "--evidence-dir",
            str(fixture.evidence_dir),
            "--parity",
            str(fixture.parity),
        ],
        cwd=SOURCE_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=20,
    )


def _write_valid_fixture(root: Path) -> VerifierFixture:
    run_id = "run-false-green-001"
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_text(evidence_dir / "workflow-green.txt", f"{run_id}\n24 passed\n")
    _write_json(
        evidence_dir / "full-e2e" / "workflow-ledger.json",
        {
            "schema_version": "workflow_e2e_ledger_v1",
            "run_id": run_id,
            "status": "succeeded",
            "workflow_ids": ["/deep-interview"],
            "ledger_paths": ["full-e2e/workflow-ledger.json"],
            "artifact_hashes": {"artifacts/workflow-state.json": "sha256:test"},
        },
    )
    _write_text(
        evidence_dir / "full-e2e" / "workflow-transcript.txt",
        f"workflow_e2e_smoke_status=succeeded\nrun_id={run_id}\n/deep-interview\n",
    )
    _write_text(evidence_dir / "artifacts" / "workflow-state.json", f'{{"run_id": "{run_id}"}}\n')
    _write_text(
        evidence_dir / "live" / "provider-events.jsonl",
        json.dumps(
            {
                "workflow": "/deep-interview",
                "provider": "openai",
                "model": "gpt-5.5",
                "provider_request_id": "req_live_001",
                "assistant_message_id": "msg_live_001",
            },
            sort_keys=True,
        )
        + "\n",
    )
    parity = root / "parity.json"
    _write_json(parity, _parity_payload())
    manifest = root / "final-manifest.json"
    _write_json(manifest, _manifest_payload(run_id))
    return VerifierFixture(evidence_dir=evidence_dir, manifest=manifest, parity=parity, run_id=run_id)


def _parity_payload() -> JsonMap:
    return {
        "schema_version": 1,
        "matrix": "gajae_workflow_parity",
        "rows": [
            {
                "id": "deep_interview_one_question_gate",
                "gajae_reference": ["gajae:test"],
                "behavioral_contract": "question gates require one runtime round",
                "asa_target": ["sim_agent/agents_sdk_runtime/workflow_deep_interview.py"],
                "tests": ["tests/test_deep_interview_runtime.py"],
                "status": "implemented",
                "verification_evidence": ["workflow-green.txt"],
                "anti_false_green": {
                    "sabotage": "remove parity evidence",
                    "detected_by": "tests/test_workflow_false_green_guards.py",
                    "blocker": "parity_evidence_missing",
                },
            }
        ],
    }


def _manifest_payload(run_id: str) -> JsonMap:
    return {
        "schema_version": "asa_workflow_gap_evidence_manifest_v1",
        "run_id": run_id,
        "parity_rows": ["deep_interview_one_question_gate"],
        "evidence_files": [{"path": "workflow-green.txt", "contains": [run_id, "24 passed"]}],
        "e2e_scenarios": [
            {
                "name": "full-workflow-loop",
                "status": "succeeded",
                "json": "full-e2e/workflow-ledger.json",
                "transcript": "full-e2e/workflow-transcript.txt",
                "run_id": run_id,
                "required_transcript": ["workflow_e2e_smoke_status=succeeded", "/deep-interview"],
                "artifacts": ["artifacts/workflow-state.json"],
                "started_at_epoch": 0,
            }
        ],
        "sabotage_cases": [{"case": "fake_stdout_no_ledger", "status": "detected", "blocker": "e2e_ledger_missing"}],
        "live_llm": [
            {
                "workflow": "/deep-interview",
                "status": "succeeded",
                "provider": "openai",
                "model": "gpt-5.5",
                "provider_request_id": "req_live_001",
                "assistant_message_id": "msg_live_001",
                "provider_events": "live/provider-events.jsonl",
            }
        ],
    }


def _remove_parity_evidence(fixture: VerifierFixture) -> None:
    (fixture.evidence_dir / "workflow-green.txt").unlink()


def _remove_negative_control(fixture: VerifierFixture) -> None:
    payload = _read_json(fixture.parity)
    rows = payload["rows"]
    rows[0].pop("anti_false_green")
    _write_json(fixture.parity, payload)


def _remove_e2e_ledger(fixture: VerifierFixture) -> None:
    (fixture.evidence_dir / "full-e2e" / "workflow-ledger.json").unlink()


def _remove_required_transcript_line(fixture: VerifierFixture) -> None:
    _write_text(fixture.evidence_dir / "full-e2e" / "workflow-transcript.txt", f"run_id={fixture.run_id}\n")


def _write_stale_artifact(fixture: VerifierFixture) -> None:
    _write_text(fixture.evidence_dir / "artifacts" / "workflow-state.json", '{"run_id": "old-run"}\n')


def _mark_sabotage_missed(fixture: VerifierFixture) -> None:
    payload = _read_json(fixture.manifest)
    payload["sabotage_cases"][0]["status"] = "missed"
    _write_json(fixture.manifest, payload)


def _mark_live_provider_fake(fixture: VerifierFixture) -> None:
    payload = _read_json(fixture.manifest)
    payload["live_llm"][0]["provider"] = "fake"
    _write_json(fixture.manifest, payload)


def _mark_live_provider_unavailable(fixture: VerifierFixture) -> None:
    payload = _read_json(fixture.manifest)
    payload["live_llm"][0] = {
        "workflow": "/deep-interview",
        "status": "blocked",
        "blocker": "live_llm_provider_unavailable",
    }
    _write_json(fixture.manifest, payload)


def _read_json(path: Path) -> JsonMap:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
