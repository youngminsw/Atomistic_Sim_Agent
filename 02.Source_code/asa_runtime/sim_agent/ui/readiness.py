from __future__ import annotations

import json
from pathlib import Path

from sim_agent.production_readiness import assess_production_readiness_from_payloads
from sim_agent.runner import OfflineRunResult
from sim_agent.schemas._parse import JsonMap, as_mapping


def offline_production_readiness_payload(result: OfflineRunResult) -> JsonMap:
    qa_report = _read_json(result.qa_report_path, "qa_report")
    report = assess_production_readiness_from_payloads(
        ledger={
            "run_id": result.run_id,
            "model_provider": {
                "provider": "controller_configured_gateway",
                "model": "gpt-5.5",
                "auth_mode": "gateway",
            },
            "md": {
                "production_ready": False,
                "hard_blockers": ["offline_demo_fixture_not_production_md"],
            },
            "remote": {
                "chain_status": "offline_demo_fixture",
                "chain_blockers": ["remote_chain_not_completed"],
            },
            "surrogate": {
                "training_gate_present": True,
                "training_gate_accepted": False,
                "training_gate_blockers": ["offline_demo_surrogate_not_production"],
            },
        },
        feature_qa_report=qa_report,
    )
    return report.payload


def _read_json(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)
