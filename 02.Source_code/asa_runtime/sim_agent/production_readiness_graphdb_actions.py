from __future__ import annotations

from sim_agent.production_readiness_contract import action_actor
from sim_agent.production_readiness_ledger import (
    SCRIPT_ROOT,
    artifact_output,
    artifact_path,
)
from sim_agent.production_readiness_missing_action import missing_action
from sim_agent.schemas._parse import JsonMap


def graphdb_apply_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    bundle_dir = artifact_path(ledger, "graphdb_import_bundle_dir")
    if not bundle_dir:
        return missing_action(action, ["graphdb_import_bundle_dir"])
    result_path = artifact_output(ledger, "graphdb_write_report.json")
    database_name = _graphdb_database_name(ledger)
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "ready_after_user_action",
        "requires_user_action": "approve_empty_neo4j_database_write",
        "command": [
            "python3",
            f"{SCRIPT_ROOT}/apply_graphdb_import_bundle.py",
            "--bundle-dir",
            bundle_dir,
            "--database-name",
            database_name,
            "--approve-write",
            "--out",
            result_path,
        ],
        "expected_artifacts": [result_path],
    }


def _graphdb_database_name(ledger: JsonMap) -> str:
    graphdb = ledger.get("graphdb")
    if isinstance(graphdb, dict):
        value = graphdb.get("database_name")
        if isinstance(value, str) and value:
            return value
    return "atomistic_sim_agent_knowledge"
