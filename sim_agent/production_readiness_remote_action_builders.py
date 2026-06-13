from __future__ import annotations

from pathlib import Path

from sim_agent.production_readiness_contract import action_actor
from sim_agent.production_readiness_ledger import (
    SCRIPT_ROOT,
    artifact_dir,
    artifact_output,
    artifact_path,
)
from sim_agent.production_readiness_missing_action import missing_action
from sim_agent.schemas._parse import JsonMap


def remote_chain_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    manifest_path = artifact_path(ledger, "remote_execution_manifest_path")
    result_path = artifact_path(ledger, "remote_chain_result_path") or artifact_output(
        ledger,
        "remote_chain_result.json",
    )
    if not manifest_path:
        return missing_action(action, ["remote_execution_manifest_path"])
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "ready_after_user_action",
        "requires_user_action": "approve_remote_or_long_compute_run",
        "command": [
            "python3",
            f"{SCRIPT_ROOT}/run_remote_chain.py",
            "--manifest",
            manifest_path,
            "--out",
            result_path,
        ],
        "expected_artifacts": [result_path],
    }


def model_endpoint_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    request_path = artifact_path(ledger, "validated_request_path")
    if not request_path:
        return missing_action(
            action,
            ["validated_request_path"],
            next_actions=_model_endpoint_recovery_steps(action, user_actions),
        )
    output_dir = str(Path(artifact_dir(ledger)) / "model_endpoint_smoke")
    payload: dict[str, object] = {
        "action": action,
        "actor": action_actor(action),
        "status": "ready_after_user_action"
        if "login_to_model_gateway_or_provide_token" in user_actions
        else "ready",
        "command": [
            "python3",
            f"{SCRIPT_ROOT}/smoke_production_gateway_client.py",
            "--request",
            request_path,
            "--output-dir",
            output_dir,
        ],
        "expected_artifacts": [
            str(Path(output_dir) / "production_gateway_smoke_ledger.json")
        ],
    }
    if "login_to_model_gateway_or_provide_token" in user_actions:
        payload["requires_user_action"] = "login_to_model_gateway_or_provide_token"
    return payload


def _model_endpoint_recovery_steps(action: str, user_actions: list[str]) -> list[str]:
    steps = ["persist_validated_request_from_agent_plan"]
    if "login_to_model_gateway_or_provide_token" in user_actions:
        steps.append("login_to_model_gateway_or_provide_token")
    steps.append(action)
    return steps
