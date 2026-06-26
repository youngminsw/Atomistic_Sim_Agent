from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sim_agent.agent_runtime.compaction_semantic import (
    build_semantic_summary_request,
    load_compaction_prompt,
    semantic_prompt_contract,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = SOURCE_ROOT / "tests" / "fixtures" / "workflow_parity" / "gajae-compaction-prompt-golden.json"
OMITTED_OLD_RAW = "GAJAE_GOLDEN_OMITTED_OLD_RAW_MUST_NOT_RETURN"
SUMMARIZED_OLD_RAW = "GAJAE_GOLDEN_SUMMARIZED_OLD_RAW"
RETAINED_PREFIX = "GAJAE_GOLDEN_RETAINED_PREFIX"
RECENT_TAIL = "GAJAE_GOLDEN_RECENT_TAIL"
CUSTOM_FOCUS = "GAJAE_GOLDEN_CUSTOM_FOCUS"
PREVIOUS_SUMMARY = "GAJAE_GOLDEN_PREVIOUS_SUMMARY"


def test_gajae_prompt_golden_system_initial_update_and_short_summary() -> None:
    golden = _load_golden()
    initial_request = _build_request(previous_summary="", additional_focus="")
    update_request = _build_request(previous_summary=PREVIOUS_SUMMARY, additional_focus=CUSTOM_FOCUS)

    assert golden["schema_version"] == 2
    assert golden["source_basis"] == "Gajae source checkout /tmp/gajae-code at 0aaccdf9b9a06defba3f17e76b37f2596f067622"
    assert "ASA current Gajae-style compaction contract" not in json.dumps(golden, sort_keys=True)
    assert golden["gajae_source"]["commit"] == "0aaccdf9b9a06defba3f17e76b37f2596f067622"
    assert golden["contract"] == _jsonable_contract(semantic_prompt_contract())
    assert golden["contract"]["prompt_sha256"] == {
        "summarization-system": "f99a409a9167a0f0d4af24e2ea0c9a015665d98a2357d5804f9661c594d9ab92",
        "compaction-summary": "60e9314cf109260c16f32a58ec01f99861166b11726daf2e69fd2a72e7b078a1",
        "compaction-update-summary": "efe34afe7f3eea58f02b2bdfdf42804f3557563c3dc7264c0a94b7055ba5f092",
        "compaction-short-summary": "f041eb5fa3b85424738a7e971581e747cf55fb34f223f5be0b29e1e8cd9b0538",
        "compaction-turn-prefix": "adc4daa19431945d9799b575b102b5edb26c04499bc0d40189bbc79978ff57f2",
        "file-operations": "061d32c633783c79c2a7eb37d5017394db8c618c8b717c50392e3ab28fda0035",
        "compaction-summary-context": "7a00a74524c0921a19ae04036a6b8f7964deea3eaf10f803d8365d10f059eea3",
    }
    assert golden["normalized_sha256"] == {
        "system": _normalized_sha256(initial_request.system_prompt),
        "initial": _normalized_sha256(initial_request.prompt),
        "update_with_previous": _normalized_sha256(update_request.prompt),
        "short_summary": _normalized_sha256(load_compaction_prompt("compaction-short-summary")),
    }
    _assert_required_parity_rows(golden)


def test_gajae_prompt_golden_additional_focus_recent_tail_and_custom_instructions() -> None:
    request = _build_request(previous_summary=PREVIOUS_SUMMARY, additional_focus=f"  {CUSTOM_FOCUS}  ")
    prompt = request.prompt

    assert CUSTOM_FOCUS in prompt
    assert PREVIOUS_SUMMARY in prompt
    assert SUMMARIZED_OLD_RAW in prompt
    assert RETAINED_PREFIX in prompt
    assert RECENT_TAIL in prompt
    assert OMITTED_OLD_RAW not in prompt
    assert request.messages_to_summarize == (
        {"role": "user", "content": SUMMARIZED_OLD_RAW, "sequence": 1},
        {"role": "assistant", "content": "summarized assistant context", "sequence": 2},
    )
    assert request.turn_prefix_messages == (
        {"role": "user", "content": RETAINED_PREFIX, "sequence": 3},
        {"role": "assistant", "content": "retained assistant bridge", "sequence": 4},
    )
    assert request.retained_messages == (
        {"role": "user", "content": RETAINED_PREFIX, "sequence": 3},
        {"role": "assistant", "content": "retained assistant bridge", "sequence": 4},
        {"role": "user", "content": RECENT_TAIL, "sequence": 5},
    )
    _assert_order(
        prompt,
        (
            "Additional focus:\n" + CUSTOM_FOCUS,
            "agent_id: golden_agent",
            "<previous-summary>\n" + PREVIOUS_SUMMARY,
            "<conversation>",
            "[1] user: " + SUMMARIZED_OLD_RAW,
            "<turn-prefix>",
            "[3] user: " + RETAINED_PREFIX,
            "<retained-tail>",
            "[5] user: " + RECENT_TAIL,
        ),
    )


def _build_request(*, previous_summary: str, additional_focus: str):
    return build_semantic_summary_request(
        agent_id="golden_agent",
        compact_id="golden-compact-001",
        compact_mode="manual",
        summary_source="manual_generated",
        messages=(
            {"role": "user", "content": OMITTED_OLD_RAW},
            {"role": "user", "content": SUMMARIZED_OLD_RAW, "sequence": 1},
            {"role": "assistant", "content": "summarized assistant context", "sequence": 2},
            {"role": "user", "content": RETAINED_PREFIX, "sequence": 3},
            {"role": "assistant", "content": "retained assistant bridge", "sequence": 4},
            {"role": "user", "content": RECENT_TAIL, "sequence": 5},
        ),
        first_kept_sequence=3,
        summary_cutoff_sequence=0,
        previous_summary=previous_summary,
        additional_focus=additional_focus,
    )


def _load_golden() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _jsonable_contract(contract: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(contract, sort_keys=True))


def _normalized_sha256(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assert_order(value: str, ordered_needles: tuple[str, ...]) -> None:
    position = -1
    for needle in ordered_needles:
        next_position = value.index(needle)
        assert next_position > position, needle
        position = next_position


def _assert_required_parity_rows(golden: dict[str, object]) -> None:
    matrix = golden["parity_matrix"]
    assert isinstance(matrix, list)
    text = json.dumps(matrix, sort_keys=True)
    for required in (
        "generateSummary",
        "prepareCompaction",
        "compact",
        "turn-prefix",
        "previous-summary",
        "short-summary",
        "file operations",
        "provider projection",
    ):
        assert required in text
    assert all(row.get("status") for row in matrix if isinstance(row, dict))
