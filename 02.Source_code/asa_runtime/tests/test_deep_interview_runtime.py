from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_deep_interview_opens_one_pending_response_schema_gate_without_evidence(tmp_path: Path) -> None:
    # Given: a deep-interview request with structured round metadata and no evidence.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    # When: the workflow harness starts the round.
    result = run_workflow_harness_smoke("deep-interview", _payload(), tmp_path)

    # Then: exactly one pending response-schema question gate is persisted.
    assert result.status == "blocked"
    assert result.gate_status == "awaiting_response"
    assert result.missing_evidence == ()
    assert result.blockers == ("workflow_gate_response_required",)
    gates = sorted((tmp_path / "deep-interview" / "gates").glob("*.json"))
    assert len(gates) == 1
    gate = json.loads(gates[0].read_text(encoding="utf-8"))
    assert gate["gate_kind"] == "response_schema"
    assert gate["status"] == "awaiting_response"
    expected_metadata = {
        "round": 1,
        "round_id": "round-1",
        "component": "requirements",
        "dimension": "scope",
        "ambiguity": 0.73,
        "question_id": "q-scope",
        "multi": False,
        "options": ["API contract", "CLI behavior"],
    }
    assert expected_metadata.items() <= gate["deep_interview"].items()
    assert not (tmp_path / "deep-interview" / "handoff.md").exists()


def test_deep_interview_prefers_structured_metadata_over_question_text(tmp_path: Path) -> None:
    # Given: conflicting text and structured metadata.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = _payload(
        deep_interview={
            "round": 2,
            "round_id": "structured-round",
            "component": "state",
            "dimension": "resume",
            "ambiguity": 0.42,
            "question_id": "q-structured",
            "multi": True,
            "options": ["Continue", "Restart"],
        },
        question="Round 9 component=wrong dimension=wrong ambiguity=0.99 question_id=q-text multi=false options=Bad",
    )

    # When: the harness starts the round.
    result = run_workflow_harness_smoke("deep-interview", payload, tmp_path)

    # Then: persisted metadata comes from payload["deep_interview"], not the text.
    gate = _gate_payload(tmp_path, result.gate)
    assert gate["deep_interview"]["round"] == 2
    assert gate["deep_interview"]["round_id"] == "structured-round"
    assert gate["deep_interview"]["component"] == "state"
    assert gate["deep_interview"]["dimension"] == "resume"
    assert gate["deep_interview"]["question_id"] == "q-structured"
    assert gate["deep_interview"]["multi"] is True


def test_deep_interview_uses_regex_metadata_fallback_when_structured_metadata_is_absent(tmp_path: Path) -> None:
    # Given: only a text question with parseable deep-interview metadata.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    payload = {
        "request_id": "regex-fallback",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-regex",
        "question": (
            "Round 3 round_id=regex-r3 component=transport dimension=auth "
            "ambiguity=0.31 question_id=q-regex multi=true options=Token|Cookie"
        ),
    }

    # When: the harness starts the round.
    result = run_workflow_harness_smoke("deep-interview", payload, tmp_path)

    # Then: regex metadata is preserved in the gate.
    gate = _gate_payload(tmp_path, result.gate)
    assert gate["deep_interview"]["round"] == 3
    assert gate["deep_interview"]["round_id"] == "regex-r3"
    assert gate["deep_interview"]["component"] == "transport"
    assert gate["deep_interview"]["dimension"] == "auth"
    assert gate["deep_interview"]["ambiguity"] == 0.31
    assert gate["deep_interview"]["question_id"] == "q-regex"
    assert gate["deep_interview"]["multi"] is True
    assert gate["deep_interview"]["options"] == ["Token", "Cookie"]


def test_deep_interview_accepts_single_multi_and_other_answers(tmp_path: Path) -> None:
    # Given: three deep-interview gates for single, multi, and Other paths.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    cases = (
        ("single", False, ["API contract", "CLI behavior"], {"selected": ["API contract"]}),
        ("multi", True, ["SQLite", "Postgres", "S3"], {"selected": ["SQLite", "S3"]}),
        ("other", False, [], {"selected": [], "other": True, "custom": "Passkeys"}),
    )

    # When: each gate receives a valid answer.
    accepted: list[str] = []
    for case_id, multi, options, answer in cases:
        root = tmp_path / case_id
        result = run_workflow_harness_smoke(
            "deep-interview",
            _payload(
                request_id=case_id,
                deep_interview={
                    "round": 1,
                    "round_id": f"{case_id}-r1",
                    "component": "requirements",
                    "dimension": case_id,
                    "ambiguity": 0.1,
                    "question_id": f"q-{case_id}",
                    "multi": multi,
                    "options": options,
                },
            ),
            root,
        )
        response = respond_workflow_gate(
            root,
            {
                "workflow_id": "deep-interview",
                "gate_id": str(result.gate["gate_id"]),
                "responder_agent_id": "orchestrator",
                "value": answer,
            },
        )
        accepted.append(response.status)

    # Then: every answer is accepted by the response-schema runtime.
    assert accepted == ["accepted", "accepted", "accepted"]


def test_deep_interview_invalid_answers_leave_gate_pending(tmp_path: Path) -> None:
    # Given: one pending single-select deep-interview gate.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    result = run_workflow_harness_smoke("deep-interview", _payload(), tmp_path)
    gate_id = str(result.gate["gate_id"])
    invalid_values = (
        "not-json-shape",
        {},
        {"selected": "API contract"},
        {"custom": "Free text without Other"},
        {"selected": []},
        {"selected": ["Nope"]},
        {"selected": ["API contract"], "extra": True},
        {"selected": ["API contract"], "other": True, "custom": "Both"},
        {"selected": ["API contract", "CLI behavior"]},
        {"selected": [], "other": True},
        {"selected": [], "other": True, "custom": "   "},
    )

    # When: malformed and semantically invalid answers are submitted.
    blockers = [
        respond_workflow_gate(
            tmp_path,
            {
                "workflow_id": "deep-interview",
                "gate_id": gate_id,
                "responder_agent_id": "orchestrator",
                "value": value,
            },
        ).blockers
        for value in invalid_values
    ]

    # Then: every invalid answer is blocked and the gate remains pending.
    assert blockers == [("workflow_gate_response_schema_mismatch",)] * len(invalid_values)
    gate = json.loads((tmp_path / "deep-interview" / "gates" / f"{gate_id}.json").read_text(encoding="utf-8"))
    assert gate["status"] == "awaiting_response"


def test_deep_interview_valid_response_appends_transcript_state_and_handoff(tmp_path: Path) -> None:
    # Given: a low-ambiguity deep-interview round.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "deep-interview",
        _payload(deep_interview=_metadata(ambiguity=0.2)),
        tmp_path,
    )

    # When: a valid answer is submitted.
    response = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "deep-interview",
            "gate_id": str(result.gate["gate_id"]),
            "responder_agent_id": "orchestrator",
            "value": {"selected": ["API contract"]},
        },
    )

    # Then: the transcript, state, and handoff are materialized from the accepted response.
    assert response.status == "accepted"
    transcript_rows = _jsonl(tmp_path / "deep-interview" / "transcript.jsonl")
    assert transcript_rows[-1]["selected_options"] == ["API contract"]
    state = json.loads((tmp_path / "deep-interview" / "state" / "round-1.json").read_text(encoding="utf-8"))
    assert state["status"] == "accepted"
    assert state["ambiguity"] == 0.2
    handoff = (tmp_path / "deep-interview" / "handoff.md").read_text(encoding="utf-8")
    assert "API contract" in handoff
    assert "ambiguity: 0.2" in handoff


def test_deep_interview_resume_does_not_duplicate_pending_gate_or_handoff(tmp_path: Path) -> None:
    # Given: an already-started deep-interview round.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    first = run_workflow_harness_smoke("deep-interview", _payload(), tmp_path)

    # When: the same workflow directory is resumed before an answer is supplied.
    second = run_workflow_harness_smoke("deep-interview", _payload(), tmp_path)

    # Then: the same pending gate is reused and no terminal artifacts are created.
    assert second.gate == first.gate
    assert len(list((tmp_path / "deep-interview" / "gates").glob("*.json"))) == 1
    assert not (tmp_path / "deep-interview" / "handoff.md").exists()


def test_deep_interview_corrupt_state_blocks_without_overwriting(tmp_path: Path) -> None:
    # Given: a corrupt persisted deep-interview state file.
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    state_dir = tmp_path / "deep-interview" / "state"
    state_dir.mkdir(parents=True)
    corrupt = state_dir / "round-1.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    # When: the workflow is started.
    result = run_workflow_harness_smoke("deep-interview", _payload(), tmp_path)

    # Then: runtime fails closed and preserves the corrupt bytes.
    assert result.status == "blocked"
    assert result.blockers == ("deep_interview_state_corrupt",)
    assert corrupt.read_text(encoding="utf-8") == "{not-json"


def test_deep_interview_max_rounds_writes_handoff_with_blocker(tmp_path: Path) -> None:
    # Given: a high-ambiguity round configured as the maximum round.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    result = run_workflow_harness_smoke(
        "deep-interview",
        _payload(deep_interview=_metadata(ambiguity=0.91), max_rounds=1),
        tmp_path,
    )

    # When: the final allowed round is answered.
    response = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "deep-interview",
            "gate_id": str(result.gate["gate_id"]),
            "responder_agent_id": "orchestrator",
            "value": {"selected": ["CLI behavior"]},
        },
    )

    # Then: the answer is accepted and the handoff records the max-round blocker.
    assert response.status == "accepted"
    handoff = (tmp_path / "deep-interview" / "handoff.md").read_text(encoding="utf-8")
    assert "deep_interview_max_rounds_reached" in handoff
    state = json.loads((tmp_path / "deep-interview" / "state" / "round-1.json").read_text(encoding="utf-8"))
    assert state["blocker"] == "deep_interview_max_rounds_reached"


def test_deep_interview_accepted_high_ambiguity_round_is_not_overwritten_on_resume(tmp_path: Path) -> None:
    # Given: a high-ambiguity round that accepts an answer but is not terminal.
    from sim_agent.agents_sdk_runtime import respond_workflow_gate, run_workflow_harness_smoke

    first = run_workflow_harness_smoke("deep-interview", _payload(deep_interview=_metadata(ambiguity=0.91)), tmp_path)
    response = respond_workflow_gate(
        tmp_path,
        {
            "workflow_id": "deep-interview",
            "gate_id": str(first.gate["gate_id"]),
            "responder_agent_id": "orchestrator",
            "value": {"selected": ["API contract"]},
        },
    )
    accepted_gate_path = tmp_path / "deep-interview" / "gates" / f"{first.gate['gate_id']}.json"
    accepted_gate = json.loads(accepted_gate_path.read_text(encoding="utf-8"))

    # When: the same round is resumed instead of providing a fresh next-round question.
    resumed = run_workflow_harness_smoke("deep-interview", _payload(deep_interview=_metadata(ambiguity=0.91)), tmp_path)

    # Then: the accepted gate and transcript are preserved; the runtime asks for a next round.
    assert response.status == "accepted"
    assert not (tmp_path / "deep-interview" / "handoff.md").exists()
    assert resumed.status == "blocked"
    assert resumed.gate_status == "accepted"
    assert resumed.blockers == ("deep_interview_next_round_required",)
    assert json.loads(accepted_gate_path.read_text(encoding="utf-8")) == accepted_gate
    assert len(_jsonl(tmp_path / "deep-interview" / "transcript.jsonl")) == 1

    # And: a distinct next-round payload opens exactly one new pending question gate.
    next_round = _metadata(ambiguity=0.63)
    next_round["round"] = 2
    next_round["round_id"] = "round-2"
    next_round["question_id"] = "q-scope-next"
    third = run_workflow_harness_smoke("deep-interview", _payload(deep_interview=next_round), tmp_path)

    assert third.gate_status == "awaiting_response"
    assert third.blockers == ("workflow_gate_response_required",)
    assert len(list((tmp_path / "deep-interview" / "gates").glob("*.json"))) == 2


def test_tui_deep_interview_outputs_round_metadata_gate_ledger_and_response_status(tmp_path: Path) -> None:
    # Given: a TUI session that starts and answers deep-interview.
    workflow_dir = tmp_path / "workflows"

    # When: the CLI runs the workflow and response commands.
    result = _run_tui(
        tmp_path,
        (
            "/workflow deep-interview --owner-agent orchestrator --target-agent orchestrator "
            "--goal-id goal-tui --deep-round 1 --deep-round-id tui-r1 --deep-component tui "
            "--deep-dimension scope --deep-ambiguity 0.2 --deep-question-id q-tui "
            f"--deep-options API,CLI --output-dir {workflow_dir}\n"
            f"/workflow-response question-q-tui '{{\"selected\":[\"API\"]}}' --workflow-id deep-interview "
            f"--responder-agent orchestrator --output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    # Then: the TUI exposes the round metadata, ledger ref, blocker, and accepted status.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_deep_interview_round=1" in result.stdout
    assert "workflow_deep_interview_round_id=tui-r1" in result.stdout
    assert "workflow_deep_interview_component=tui" in result.stdout
    assert "workflow_deep_interview_dimension=scope" in result.stdout
    assert "workflow_deep_interview_ambiguity=0.2" in result.stdout
    assert "workflow_gate_id=question-q-tui" in result.stdout
    assert "workflow_gate_ledger_ref=deep-interview/gates/question-q-tui.json" in result.stdout
    assert "workflow_blocker=workflow_gate_response_required" in result.stdout
    assert "workflow_response_status=accepted" in result.stdout


def _payload(
    *,
    request_id: str = "deep-runtime",
    deep_interview: dict[str, object] | None = None,
    question: str = "Which surface should be clarified first?",
    max_rounds: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": request_id,
        "user_goal": "Clarify runtime requirements",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-deep",
        "question": question,
        "deep_interview": deep_interview or _metadata(),
    }
    if max_rounds is not None:
        payload["max_rounds"] = max_rounds
    return payload


def _metadata(*, ambiguity: float = 0.73) -> dict[str, object]:
    return {
        "round": 1,
        "round_id": "round-1",
        "component": "requirements",
        "dimension": "scope",
        "ambiguity": ambiguity,
        "question_id": "q-scope",
        "multi": False,
        "options": ["API contract", "CLI behavior"],
    }


def _gate_payload(output_dir: Path, gate: dict[str, object] | None) -> dict[str, object]:
    assert gate is not None
    ledger_ref = gate["ledger_ref"]
    assert isinstance(ledger_ref, str)
    return json.loads((output_dir / ledger_ref).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
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
