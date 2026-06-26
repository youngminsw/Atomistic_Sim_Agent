from __future__ import annotations

import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = SOURCE_ROOT / "tests" / "fixtures" / "workflow_parity"
JSON_MATRIX = MATRIX_DIR / "gajae-workflow-parity-matrix.json"
MARKDOWN_MATRIX = MATRIX_DIR / "gajae-workflow-parity-matrix.md"

REQUIRED_FIELDS = {
    "id",
    "gajae_reference",
    "behavioral_contract",
    "asa_target",
    "tests",
    "status",
    "verification_evidence",
}
REQUIRED_ROW_IDS = {
    "workflow_gate_flat_shape",
    "wrapped_event_envelope",
    "ralplan_approval_gate_parsing",
    "response_validation_enum_rejection_acceptance",
    "deep_interview_one_question_gate",
    "action_lifecycle_first_valid_resolution",
    "task_goal_context_summaries",
    "ultragoal_signoff_checkpoint_gate",
    "visual_qa_evidence_workflow",
    "ultraresearch_insane_search_evidence_boundary",
    "tui_tool_persistence_coverage",
}
ALLOWED_STATUSES = {"implemented"}
GENERIC_VERIFICATION_EVIDENCE = [".omo/ulw-loop/evidence/G002-final-workflow-pytest.log"]
ROW_VERIFICATION_EVIDENCE = {
    "deep_interview_one_question_gate": [
        ".omo/evidence/asa-gajae-workflow-gap-closure/t05-deep-interview-red.txt",
        ".omo/evidence/asa-gajae-workflow-gap-closure/t05-deep-interview-green.txt",
        ".omo/evidence/asa-gajae-workflow-gap-closure/t05-deep-interview-transcript.txt",
    ],
    "visual_qa_evidence_workflow": [
        ".omo/evidence/asa-gajae-workflow-gap-closure/t08-visual-qa-green.txt",
        ".omo/evidence/asa-gajae-workflow-gap-closure/t08-visual-qa-transcript.txt",
    ],
    "ultraresearch_insane_search_evidence_boundary": [
        ".omo/evidence/asa-gajae-workflow-gap-closure/t09-ultraresearch-green.txt",
        ".omo/evidence/asa-gajae-workflow-gap-closure/t09-ultraresearch-transcript.txt",
    ],
}


def test_gajae_workflow_parity_matrix_has_required_rows_and_fields() -> None:
    data = json.loads(JSON_MATRIX.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["matrix"] == "gajae_workflow_parity"
    assert isinstance(data["rows"], list)

    rows_by_id = {row["id"]: row for row in data["rows"]}
    assert REQUIRED_ROW_IDS <= set(rows_by_id)

    for row_id in REQUIRED_ROW_IDS:
        row = rows_by_id[row_id]
        assert REQUIRED_FIELDS <= set(row), row_id
        assert row["id"] == row_id
        assert row["status"] in ALLOWED_STATUSES
        assert row["gajae_reference"], row_id
        assert row["behavioral_contract"], row_id
        assert row["asa_target"], row_id
        assert row["tests"], row_id
        assert row["verification_evidence"], row_id
        assert not any("planned:" in item for item in row["asa_target"]), row_id
        assert not any("planned:" in item for item in row["tests"]), row_id
        assert not any("planned:" in item for item in row["verification_evidence"]), row_id
        assert row["verification_evidence"] == ROW_VERIFICATION_EVIDENCE.get(
            row_id,
            GENERIC_VERIFICATION_EVIDENCE,
        ), row_id


def test_gajae_workflow_parity_matrix_markdown_mentions_every_required_row() -> None:
    matrix_text = MARKDOWN_MATRIX.read_text(encoding="utf-8")

    for row_id in REQUIRED_ROW_IDS:
        assert row_id in matrix_text
