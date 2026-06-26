from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_compaction_smoke_cli_writes_compaction_resume_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"

    result, matrix, transcript, e2e = _run_and_load_smoke(output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "compaction_smoke=true" in result.stdout
    assert matrix["status"] == "succeeded"
    assert matrix["blockers"] == []
    assert matrix["checks"]["manual_compact_replayed"] is True
    assert matrix["checks"]["manual_compact_no_user_summary"] is True
    assert matrix["checks"]["manual_compact_semantic_summary_source_llm"] is True
    assert matrix["checks"]["auto_threshold_compacted"] is True
    assert matrix["checks"]["poison_blocked"] is True
    assert matrix["checks"]["stale_cursor_blocked"] is True
    assert matrix["checks"]["orphan_tool_result_blocked"] is True
    assert matrix["checks"]["invalid_state_blocked_before_provider"] is True
    assert matrix["checks"]["invalid_state_prompt_manifest_written_false"] is True
    assert matrix["checks"]["invalid_state_gateway_post_called_false"] is True
    assert matrix["checks"]["prompt_manifest_has_compact_summary_layer"] is True
    assert matrix["checks"]["prompt_manifest_has_validated_summary"] is True
    assert matrix["checks"]["old_raw_retained_on_disk"] is True
    assert matrix["checks"]["multi_old_raw_retained_on_disk"] is True
    assert matrix["checks"]["old_raw_absent_from_prompt_manifest"] is True
    assert matrix["checks"]["multi_old_raw_absent_from_prompt_manifest"] is True
    assert matrix["checks"]["old_raw_absent_from_provider_protocol"] is True
    assert matrix["checks"]["multi_old_raw_absent_from_provider_protocol"] is True
    assert matrix["checks"]["tail_visible_in_prompt_manifest"] is True
    assert matrix["checks"]["current_turn_visible_in_prompt_manifest"] is True
    assert matrix["checks"]["summary_visible_in_provider_protocol"] is True
    assert matrix["checks"]["tail_visible_in_provider_protocol"] is True
    assert matrix["checks"]["current_turn_visible_in_provider_protocol"] is True
    assert matrix["checks"]["provider_shape_keys_present"] is True
    assert matrix["checks"]["auto_threshold_accounting_present"] is True
    assert matrix["poison"]["blocker"] == "compact_summary_poisoned"
    assert matrix["stale_cursor"]["blocker"] == "stale_compact_cursor"
    assert matrix["orphan_tool_result"]["blocker"] == "orphan_tool_result"
    assert matrix["resume"]["opened_as"] == "resumed"
    assert matrix["resume"]["turn"]["status"] == "succeeded"
    assert "compact_replay_status=replayed" in transcript
    assert "prompt_manifest_layer_kinds=" in transcript
    assert "invalid_state_gateway_post_called=False" in transcript
    assert e2e["status"] == "succeeded"
    assert e2e["provider_prompt_manifest"]["has_compact_summary_layer"] is True
    assert e2e["provider_protocol"]["old_raw_absent"] is True


def test_smoke_evidence_contains_provider_shape_keys_old_raw_absence_and_append_only_hashes(tmp_path: Path) -> None:
    _result, matrix, _transcript, e2e = _run_and_load_smoke(tmp_path / "evidence")

    assert matrix["provider_shape_keys"] == {
        "openai_responses": ["instructions", "input", "tools"],
        "openai_chat_completions": ["messages", "tools"],
        "anthropic_messages": ["system", "messages", "tools"],
        "gemini_generate_content": ["systemInstruction", "contents", "tools"],
    }
    assert all(item["required_keys_present"] for item in matrix["provider_shape_key_evidence"].values())
    assert e2e["provider_shape_key_evidence"] == matrix["provider_shape_key_evidence"]
    auto = matrix["auto"]
    assert auto["context_window_tokens"] > 0
    assert auto["threshold_tokens"] > 0
    assert auto["estimated_context_tokens"] > auto["threshold_tokens"]
    assert auto["threshold_crossed"] is True
    assert matrix["checks"]["multi_old_raw_retained_on_disk"] is True
    assert matrix["checks"]["multi_old_raw_absent_from_prompt_manifest"] is True
    assert matrix["checks"]["multi_old_raw_absent_from_provider_protocol"] is True
    append_only = matrix["append_only_message_log"]
    assert append_only["compaction_preserved_messages_file"] is True
    assert append_only["final_is_append_only_growth"] is True
    assert append_only["final_line_count"] >= append_only["after_compaction_line_count"]
    assert all(proof["retained_on_disk"] for proof in append_only["old_raw_hashes"])
    assert all(proof["sha256"] and "SMOKE_OLD_RAW" not in proof["sha256"] for proof in append_only["old_raw_hashes"])
    assert e2e["append_only_message_log"]["final_sha256"] == append_only["final_sha256"]


def test_smoke_evidence_records_invalid_state_no_manifest_no_post(tmp_path: Path) -> None:
    _result, matrix, _transcript, e2e = _run_and_load_smoke(tmp_path / "evidence")

    invalid = matrix["invalid_provider_boundary"]
    assert invalid["blocked"] is True
    assert invalid["status"] == "blocked"
    assert invalid["blockers"] == ["compact_summary_poisoned"]
    assert invalid["prompt_manifest_written"] is False
    assert invalid["gateway_post_called"] is False
    assert matrix["checks"]["invalid_state_prompt_manifest_written_false"] is True
    assert matrix["checks"]["invalid_state_gateway_post_called_false"] is True
    assert e2e["invalid_provider_boundary"] == invalid


def test_smoke_evidence_records_manual_compact_semantic_summary_and_recent_tail(tmp_path: Path) -> None:
    _result, matrix, _transcript, e2e = _run_and_load_smoke(tmp_path / "evidence")

    manual = matrix["manual"]
    assert manual["no_user_supplied_summary"] is True
    assert manual["semantic_summary_source"] == "llm_semantic"
    assert manual["summary_contains_validated_sentinel"] is True
    assert matrix["checks"]["tail_visible_in_prompt_manifest"] is True
    assert matrix["checks"]["current_turn_visible_in_prompt_manifest"] is True
    assert matrix["checks"]["multi_old_raw_absent_from_provider_protocol"] is True
    assert e2e["compaction"]["semantic_summary_source"] == "llm_semantic"


def _run_and_load_smoke(output_dir: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object], str, dict[str, object]]:
    result = _run_compaction_smoke(output_dir)
    matrix_path = output_dir / "task-9-compaction-parity-matrix.json"
    transcript_path = output_dir / "task-9-compaction.txt"
    e2e_path = output_dir / "final-f3-e2e.json"
    assert result.returncode == 0, result.stdout + result.stderr
    assert matrix_path.is_file()
    assert transcript_path.is_file()
    assert e2e_path.is_file()
    return (
        result,
        json.loads(matrix_path.read_text(encoding="utf-8")),
        transcript_path.read_text(encoding="utf-8"),
        json.loads(e2e_path.read_text(encoding="utf-8")),
    )


def _run_compaction_smoke(output_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sim_agent.cli.main",
            "--compaction-smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=90,
    )
