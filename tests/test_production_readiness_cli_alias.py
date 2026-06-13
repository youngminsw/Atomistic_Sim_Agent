from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_production_readiness_report_accepts_graphdb_write_report_alias(tmp_path: Path) -> None:
    ledger_path = tmp_path / "agent_run_ledger.json"
    endpoint_path = tmp_path / "endpoint_smoke.json"
    graphdb_path = tmp_path / "graphdb_write_report.json"
    feature_path = tmp_path / "feature_qa.json"
    readiness_path = tmp_path / "readiness.json"
    ledger_path.write_text(json.dumps(_complete_ledger()), encoding="utf-8")
    endpoint_path.write_text(json.dumps({"ok": True, "real_endpoint_called": True}), encoding="utf-8")
    graphdb_path.write_text(json.dumps(_graphdb_write_report()), encoding="utf-8")
    feature_path.write_text(json.dumps({"status": "pass", "evidence_scope": "production_run"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "report_production_readiness.py"),
            "--ledger",
            str(ledger_path),
            "--model-endpoint-smoke-report",
            str(endpoint_path),
            "--graphdb-write-report",
            str(graphdb_path),
            "--feature-qa-report",
            str(feature_path),
            "--out",
            str(readiness_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["production_ready"] is True
    assert "graphdb_live_ingest_passed" in report["evidence"]


def _complete_ledger() -> dict[str, object]:
    return {
        "run_id": "complete-run",
        "model_provider": {
            "provider": "oauth_gateway",
            "model": "gpt-5.5",
            "auth_mode": "gateway",
        },
        "md": {
            "production_ready": True,
            "hard_blockers": [],
        },
        "remote": {
            "chain_status": "remote_chain_completed",
            "chain_blockers": [],
        },
        "surrogate": {
            "training_gate_present": True,
            "training_gate_accepted": True,
            "training_gate_blockers": [],
        },
    }


def _graphdb_write_report() -> dict[str, object]:
    return {
        "applied": True,
        "status": "applied",
        "blocker_reasons": [],
        "database_name": "atomistic_sim_agent_knowledge",
        "row_counts": {
            "sources": 6,
            "understandings": 6,
            "claims": 6,
            "entities": 18,
        },
        "executed_statement_kinds": [
            "constraint",
            "sources",
            "understandings",
            "claims",
            "entities",
        ],
    }
