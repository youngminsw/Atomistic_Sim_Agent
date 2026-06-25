from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import assert_never


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agent_harness import (
    AgentPlanArtifactError,
    OfflineModelClient,
    RunStatus,
    SimulationAgentHarness,
    write_agent_plan_artifacts,
)
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.model_provider_payload import model_provider_payload
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    try:
        payload = as_mapping(json.loads(Path(args.request).read_text(encoding="utf-8")), "simulation_request")
        endpoint = ModelProviderConfig.from_mapping(model_provider_payload(payload))
        result = SimulationAgentHarness(endpoint=endpoint, client=OfflineModelClient()).plan(payload)
        output_dir = _optional_path(args.output_dir)
    except (json.JSONDecodeError, OSError, SchemaValidationError, ProviderConfigPolicyError, ModelPolicyError) as exc:
        print(str(exc))
        return 1

    match result.status:
        case RunStatus.CLARIFICATION_REQUIRED:
            print("clarification_required=true")
            if result.clarification is not None:
                print(f"missing_fields={','.join(result.clarification.missing_fields)}")
            print(result.final_output)
            return 0
        case RunStatus.PLANNED:
            print("planned=true")
            print(f"run_id={result.run_id}")
            print(f"artifacts={len(result.artifacts)}")
            if output_dir is not None:
                try:
                    bundle = write_agent_plan_artifacts(output_dir, payload, result)
                except AgentPlanArtifactError as exc:
                    print(str(exc))
                    return 1
                print(f"artifact_manifest_path={bundle.manifest_path}")
                print(f"md_campaign_plan_path={bundle.md_campaign_plan_path}")
                print(f"validated_request_path={bundle.validated_request_path}")
            return 0
        case RunStatus.BLOCKED:
            print(result.final_output)
            return 1
        case unreachable:
            assert_never(unreachable)


def _optional_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw)


if __name__ == "__main__":
    raise SystemExit(main())
