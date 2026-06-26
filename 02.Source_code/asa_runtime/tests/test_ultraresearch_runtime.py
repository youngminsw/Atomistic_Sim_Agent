from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_ultraresearch_materializes_public_source_evidence_boundary(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("ultraresearch", _payload(), tmp_path)

    assert result.status == "ready"
    assert result.gate_status == "passed"
    assert result.artifact_refs == (
        "ultraresearch/acquisition-plan.json",
        "ultraresearch/research-journal.jsonl",
        "ultraresearch/source-ledger.jsonl",
        "ultraresearch/expansion-log.md",
        "ultraresearch/synthesis-checkpoint.md",
    )
    plan = _read_json(tmp_path / "ultraresearch" / "acquisition-plan.json")
    source = _jsonl(tmp_path / "ultraresearch" / "source-ledger.jsonl")[0]
    synthesis = (tmp_path / "ultraresearch" / "synthesis-checkpoint.md").read_text(encoding="utf-8")
    assert plan["schema_version"] == "ultraresearch_acquisition_v1"
    assert plan["insane_search"]["skill_id"] == "insane_search"
    assert plan["insane_search"]["grid_exhausted"] is False
    assert plan["insane_search"]["must_invoke_playwright_mcp"] is False
    assert source["content_trust"] == "untrusted_evidence"
    assert source["model_instruction_allowed"] is False
    assert source["url"] == "https://example.com/public/asa-workflows"
    assert "citation_required: true" in synthesis


def test_ultraresearch_blocks_private_source_url(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    result = run_workflow_harness_smoke("ultraresearch", _payload(url="http://127.0.0.1/internal"), tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("insane_search_public_content_only",)
    assert not (tmp_path / "ultraresearch" / "source-ledger.jsonl").exists()


def test_ultraresearch_blocks_credentialed_or_paywalled_trace(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    trace = _trace(auth_required=True, stop_reason="auth_required")
    result = run_workflow_harness_smoke("ultraresearch", _payload(trace=trace), tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ultraresearch_credentialed_source_denied",)


def test_ultraresearch_blocks_failed_trace_until_insane_search_is_exhausted(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    trace = _trace(ok=False, grid_exhausted=False, untried_routes=["playwright_mcp"], must_invoke_playwright_mcp=True)
    result = run_workflow_harness_smoke("ultraresearch", _payload(trace=trace), tmp_path)

    assert result.status == "blocked"
    assert result.gate_status == "blocked"
    assert result.blockers == ("ultraresearch_acquisition_not_exhausted",)


def test_ultraresearch_artifacts_are_idempotent_and_block_tamper(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke

    first = run_workflow_harness_smoke("ultraresearch", _payload(), tmp_path)
    plan = tmp_path / "ultraresearch" / "acquisition-plan.json"
    plan.write_text('{"tampered": true}\n', encoding="utf-8")

    second = run_workflow_harness_smoke("ultraresearch", _payload(), tmp_path)

    assert first.status == "ready"
    assert second.status == "blocked"
    assert second.gate_status == "blocked"
    assert second.blockers == ("ultraresearch_artifact_conflict",)


def test_tui_ultraresearch_accepts_structured_insane_search_trace(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflows"
    trace = json.dumps(_trace(), sort_keys=True)
    result = _run_tui(
        tmp_path,
        (
            "/workflow ultraresearch --owner-agent orchestrator --target-agent orchestrator --goal-id goal-research "
            "--research-question 'How does ASA prove workflow parity?' "
            "--source-journal journals/asa-workflows.jsonl "
            f"--insane-search-trace '{trace}' "
            f"--output-dir {workflow_dir}\n"
            "/exit\n"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow=ultraresearch" in result.stdout
    assert "workflow_status=ready" in result.stdout
    assert (
        "workflow_artifact_refs=ultraresearch/acquisition-plan.json,ultraresearch/research-journal.jsonl,"
        "ultraresearch/source-ledger.jsonl,ultraresearch/expansion-log.md,ultraresearch/synthesis-checkpoint.md"
        in result.stdout
    )
    source = _jsonl(workflow_dir / "ultraresearch" / "source-ledger.jsonl")[0]
    assert source["model_instruction_allowed"] is False


def _payload(*, trace: JsonMap | None = None, url: str = "https://example.com/public/asa-workflows") -> JsonMap:
    return {
        "request_id": "ultraresearch-rich",
        "user_goal": "Research ASA workflow parity.",
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": "goal-ultraresearch-rich",
        "evidence": {
            "research_question": "How does ASA prove workflow parity?",
            "source_journal": "journals/asa-workflows.jsonl",
            "insane_search_trace": trace if trace is not None else _trace(url=url),
        },
    }


def _trace(
    *,
    ok: bool = True,
    grid_exhausted: bool = False,
    untried_routes: list[str] | None = None,
    must_invoke_playwright_mcp: bool = False,
    auth_required: bool = False,
    stop_reason: str = "success",
    url: str = "https://example.com/public/asa-workflows",
) -> JsonMap:
    return {
        "skill_id": "insane_search",
        "surface": "skill",
        "ok": ok,
        "public_only": True,
        "ssrf_safe": True,
        "auth_required": auth_required,
        "grid_exhausted": grid_exhausted,
        "untried_routes": [] if untried_routes is None else untried_routes,
        "must_invoke_playwright_mcp": must_invoke_playwright_mcp,
        "stop_reason": stop_reason,
        "routes": ["phase0", "fetch_chain"],
        "trace": [
            {
                "phase": "probe",
                "executor": "curl_cffi",
                "url": url,
                "status": 200,
                "verdict": "weak_ok",
                "reasons": ["public fixture"],
            }
        ],
        "sources": [
            {
                "url": url,
                "route": "fetch_chain",
                "title": "ASA workflow public evidence",
                "evidence_ref": "trace[0]",
            }
        ],
    }


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
