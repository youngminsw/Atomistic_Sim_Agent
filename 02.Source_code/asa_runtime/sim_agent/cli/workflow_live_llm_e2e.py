from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.schemas._parse import JsonMap


LIVE_LLM_BLOCKER: Final = "live_llm_provider_unavailable"


@dataclass(frozen=True, slots=True)
class WorkflowLiveLlmE2ERequest:
    output_dir: Path
    scenario: str


@dataclass(frozen=True, slots=True)
class WorkflowLiveLlmE2EResult:
    status: str
    output_json: Path
    provider_events_path: Path
    blockers: tuple[str, ...]


def run_workflow_live_llm_e2e(request: WorkflowLiveLlmE2ERequest) -> WorkflowLiveLlmE2EResult:
    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(request.scenario)
    provider_events_path = output_dir / "provider-events.jsonl"
    live_row: JsonMap = {
        "workflow": "/deep-interview",
        "status": "blocked",
        "blocker": LIVE_LLM_BLOCKER,
        "run_id": run_id,
    }
    provider_events_path.write_text(json.dumps(live_row, sort_keys=True) + "\n", encoding="utf-8")
    payload: JsonMap = {
        "schema_version": "asa_workflow_live_llm_e2e_v1",
        "scenario": request.scenario,
        "run_id": run_id,
        "status": "blocked",
        "blockers": [LIVE_LLM_BLOCKER],
        "live_llm": [live_row],
        "provider_events": provider_events_path.name,
        "requires_env": {"ASA_LIVE_LLM_E2E": "1"},
        "live_opt_in": os.environ.get("ASA_LIVE_LLM_E2E") == "1",
    }
    output_json = output_dir / "workflow-live-llm.json"
    _write_json(output_json, payload)
    return WorkflowLiveLlmE2EResult("blocked", output_json, provider_events_path, (LIVE_LLM_BLOCKER,))


def _run_id(scenario: str) -> str:
    safe_scenario = scenario.replace("_", "-").replace("/", "-")
    return f"{safe_scenario}-{int(time.time() * 1000)}"


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
