from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime import (
    AgentsSdkRuntimeError,
    run_agent_team_session_smoke,
    run_agents_sdk_runtime_dry_run,
    write_agent_team_session_ledger,
    write_agents_sdk_runtime_ledger,
)
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-sdk-smoke", action="store_true")
    parser.add_argument("--team-session-smoke", action="store_true")
    parser.add_argument("--simulate-agent-failure")
    parser.add_argument("--slurm-job-script", action="store_true")
    parser.add_argument("--qa-job-script-reviewed", action="store_true")
    args = parser.parse_args()

    try:
        payload = as_mapping(json.loads(Path(args.request).read_text(encoding="utf-8")), "simulation_request")
        endpoint = ModelProviderConfig.from_mapping(as_mapping(payload.get("llm_endpoint"), "llm_endpoint"))
        result = run_agents_sdk_runtime_dry_run(payload, endpoint, run_sdk_smoke=args.run_sdk_smoke)
        ledger_path = write_agents_sdk_runtime_ledger(Path(args.output_dir), result)
        team_result = None
        team_ledger_path = None
        if args.team_session_smoke:
            team_result = run_agent_team_session_smoke(
                payload,
                endpoint,
                Path(args.output_dir),
                simulate_agent_failure=args.simulate_agent_failure,
                slurm_job_script=args.slurm_job_script,
                qa_job_script_reviewed=args.qa_job_script_reviewed,
            )
            team_ledger_path = write_agent_team_session_ledger(Path(args.output_dir), team_result)
    except (
        AgentsSdkRuntimeError,
        json.JSONDecodeError,
        OSError,
        SchemaValidationError,
        ProviderConfigPolicyError,
        ModelPolicyError,
    ) as exc:
        print(str(exc))
        return 1

    print(f"agents_sdk_runtime_ledger_path={ledger_path}")
    print(f"sdk_available={str(result.sdk_available).lower()}")
    print(f"sdk_run_completed={str(result.sdk_run_completed).lower()}")
    print(f"handoffs={','.join(result.handoff_sequence)}")
    print(f"approval_required={str(any(gate.status.value == 'required' for gate in result.approval_gates)).lower()}")
    if team_result is not None and team_ledger_path is not None:
        print(f"agent_team_session_ledger_path={team_ledger_path}")
        print(f"team_status={team_result.status.value}")
        print(f"deadlock={str(team_result.deadlock).lower()}")
        for blocker in team_result.hard_blockers:
            print(f"hard_blocker={blocker}")
        for event in team_result.recoverable_events:
            if event.startswith("agent_failure:"):
                print(f"agent_failure={event.split(':', 1)[1]}")
    if team_result is not None and not team_result.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
