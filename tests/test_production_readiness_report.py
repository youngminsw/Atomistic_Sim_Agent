from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_production_readiness_report_blocks_planned_only_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "agent_run_ledger.json"
    ledger_path.write_text(json.dumps(_planned_ledger()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "report_production_readiness.py"),
            "--ledger",
            str(ledger_path),
            "--out",
            str(tmp_path / "readiness.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert "production_ready=false" in result.stdout
    assert "user_action_required=true" in result.stdout
    assert report["production_ready"] is False
    assert "amorphous_lammps_structure_source_required" in report["hard_blockers"]
    assert "remote_chain_not_completed" in report["hard_blockers"]
    assert "surrogate_training_gate_not_accepted" in report["hard_blockers"]
    assert "model_endpoint_smoke_required" in report["hard_blockers"]
    assert "graphdb_live_ingest_required" in report["hard_blockers"]
    assert "production_feature_scale_report_required" in report["hard_blockers"]
    assert "login_to_model_gateway_or_provide_token" in report["user_actions"]
    assert "approve_remote_or_long_compute_run" in report["user_actions"]
    assert "approve_empty_neo4j_database_write" in report["user_actions"]
    assert "resolve_md_production_blockers" in report["agent_actions"]


def test_production_readiness_report_accepts_full_external_evidence(tmp_path: Path) -> None:
    ledger_path = tmp_path / "agent_run_ledger.json"
    endpoint_path = tmp_path / "endpoint_smoke.json"
    graphdb_path = tmp_path / "graphdb_ingest.json"
    feature_path = tmp_path / "feature_qa.json"
    ledger_path.write_text(json.dumps(_complete_ledger()), encoding="utf-8")
    endpoint_path.write_text(json.dumps({"ok": True, "real_endpoint_called": True}), encoding="utf-8")
    graphdb_path.write_text(json.dumps({"accepted": True, "live_write_completed": True}), encoding="utf-8")
    feature_path.write_text(json.dumps({"status": "pass", "evidence_scope": "production_run"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "report_production_readiness.py"),
            "--ledger",
            str(ledger_path),
            "--model-endpoint-smoke-report",
            str(endpoint_path),
            "--graphdb-ingest-report",
            str(graphdb_path),
            "--feature-qa-report",
            str(feature_path),
            "--out",
            str(tmp_path / "readiness.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "production_ready=true" in result.stdout
    assert report["production_ready"] is True
    assert report["hard_blockers"] == []
    assert report["user_actions"] == []
    assert "md_production_ready" in report["evidence"]
    assert "model_endpoint_smoke_passed" in report["evidence"]
    assert "graphdb_live_ingest_passed" in report["evidence"]
    assert "production_feature_scale_passed" in report["evidence"]


def test_production_readiness_report_accepts_gateway_smoke_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "agent_run_ledger.json"
    endpoint_path = tmp_path / "production_gateway_smoke_ledger.json"
    graphdb_path = tmp_path / "graphdb_ingest.json"
    feature_path = tmp_path / "feature_qa.json"
    ledger_path.write_text(json.dumps(_complete_ledger()), encoding="utf-8")
    endpoint_path.write_text(json.dumps(_gateway_smoke_ledger()), encoding="utf-8")
    graphdb_path.write_text(json.dumps({"accepted": True, "live_write_completed": True}), encoding="utf-8")
    feature_path.write_text(json.dumps({"status": "pass", "evidence_scope": "production_run"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "report_production_readiness.py"),
            "--ledger",
            str(ledger_path),
            "--model-endpoint-smoke-report",
            str(endpoint_path),
            "--graphdb-ingest-report",
            str(graphdb_path),
            "--feature-qa-report",
            str(feature_path),
            "--out",
            str(tmp_path / "readiness.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["production_ready"] is True
    assert "model_endpoint_smoke_passed" in report["evidence"]


def test_production_readiness_report_accepts_live_graphdb_write_report(tmp_path: Path) -> None:
    ledger_path = tmp_path / "agent_run_ledger.json"
    endpoint_path = tmp_path / "endpoint_smoke.json"
    graphdb_path = tmp_path / "graphdb_write_report.json"
    feature_path = tmp_path / "feature_qa.json"
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
            "--graphdb-ingest-report",
            str(graphdb_path),
            "--feature-qa-report",
            str(feature_path),
            "--out",
            str(tmp_path / "readiness.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    report = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["production_ready"] is True
    assert "graphdb_live_ingest_passed" in report["evidence"]


def _planned_ledger() -> dict[str, object]:
    return {
        "run_id": "planned-run",
        "model_provider": {
            "provider": "openclaw",
            "model": "gpt-5.5",
            "auth_mode": "oauth",
        },
        "md": {
            "production_ready": False,
            "hard_blockers": ["amorphous_lammps_structure_source_required"],
        },
        "remote": {
            "chain_status": "",
            "chain_blockers": [],
        },
        "surrogate": {
            "training_gate_present": False,
            "training_gate_accepted": None,
        },
    }


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


def _gateway_smoke_ledger() -> dict[str, object]:
    return {
        "ledger_version": "production_gateway_smoke_v1",
        "production_smoke": True,
        "offline": False,
        "fake_gateway_model": False,
        "provider": "oauth_gateway",
        "model": "gpt-5.5",
        "auth_mode": "gateway",
        "base_url": "http://127.0.0.1:9999/v1",
        "gateway_health_ok": True,
        "models_count": 1,
        "endpoint_status": 200,
        "gateway_request_id": "gw-python-smoke",
        "hard_blockers": [],
        "session_files": ["orchestrator.jsonl"],
        "final_output": "production_gateway_client_ready",
    }


def _graphdb_write_report() -> dict[str, object]:
    return {
        "applied": True,
        "status": "applied",
        "blocker_reasons": [],
        "database_name": "atomistic_sim_agent_knowledge",
        "row_counts": {
            "sources": 9,
            "understandings": 9,
            "claims": 9,
            "entities": 28,
        },
        "executed_statement_kinds": [
            "constraint",
            "constraint",
            "constraint",
            "constraint",
            "sources",
            "understandings",
            "claims",
            "entities",
        ],
    }
