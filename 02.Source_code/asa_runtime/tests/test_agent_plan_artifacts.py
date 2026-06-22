from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
REQUEST_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


def _load_request(name: str) -> JsonMap:
    return as_mapping(json.loads((REQUEST_ROOT / name).read_text(encoding="utf-8")), name)


def test_write_agent_plan_artifacts_persists_md_campaign_payload(tmp_path: Path) -> None:
    from sim_agent.agent_harness import (
        OfflineModelClient,
        SimulationAgentHarness,
        write_agent_plan_artifacts,
    )
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = _load_request("valid_ar_si_pr_hole.json")
    endpoint = ModelProviderConfig.from_mapping(as_mapping(payload["llm_endpoint"], "llm_endpoint"))
    result = SimulationAgentHarness(endpoint=endpoint, client=OfflineModelClient()).plan(payload)

    bundle = write_agent_plan_artifacts(tmp_path, payload, result)

    manifest = as_mapping(json.loads(bundle.manifest_path.read_text(encoding="utf-8")), "manifest")
    campaign = as_mapping(json.loads(bundle.md_campaign_plan_path.read_text(encoding="utf-8")), "campaign")
    request = as_mapping(json.loads(bundle.validated_request_path.read_text(encoding="utf-8")), "request")
    assert manifest["run_id"] == "plan-valid_ar_si_pr_hole"
    assert manifest["artifact_types"] == ["md_campaign_plan", "run_manifest", "validated_request"]
    assert campaign["protocol_id"] == "continuous_stratified_bombardment"
    assert campaign["material_id"] == "Si"
    assert request["request_id"] == "valid_ar_si_pr_hole"
    assert bundle.artifact_count == 3


def test_smoke_agent_plan_cli_writes_plan_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "smoke_agent_plan.py"),
            "--offline",
            "--request",
            str(REQUEST_ROOT / "valid_ar_si_pr_hole.json"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest = as_mapping(json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")), "manifest")
    campaign = as_mapping(json.loads((tmp_path / "md_campaign_plan.json").read_text(encoding="utf-8")), "campaign")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "artifact_manifest_path=" in result.stdout
    assert manifest["run_id"] == "plan-valid_ar_si_pr_hole"
    assert campaign["ion_species"] == "Ar"
