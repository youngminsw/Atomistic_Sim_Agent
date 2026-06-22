from __future__ import annotations

from collections.abc import Mapping

from sim_agent.schemas._parse import JsonMap, as_mapping

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
MutableJsonMap = dict[str, JsonValue]


def amorphous_blocked_ledger() -> MutableJsonMap:
    return {
        "run_id": "blocked-amorphous-run",
        "artifact_dir": "/tmp/run",
        "compute_target": {
            "host": "gpu-5090",
            "environment_name": "atomistic-sim-gpu",
            "ssh_target": "swym@10.24.12.85",
            "ssh_port": 55555,
        },
        "model_provider": {"auth_mode": "oauth"},
        "md": {
            "production_ready": False,
            "hard_blockers": ["amorphous_lammps_structure_source_required"],
        },
        "remote": {"chain_status": "", "chain_blockers": []},
        "surrogate": {"training_gate_accepted": False},
    }


def actions_by_id(payload: JsonMap) -> dict[str, JsonMap]:
    action_plan = payload.get("action_plan")
    assert isinstance(action_plan, list)
    actions: dict[str, JsonMap] = {}
    for action in action_plan:
        if not isinstance(action, Mapping):
            continue
        action_map = as_mapping(action, "action_plan item")
        action_id = action_map.get("action")
        if isinstance(action_id, str):
            actions[action_id] = action_map
    return actions


def string_list(payload: JsonMap, field: str) -> list[str]:
    value = payload.get(field)
    assert isinstance(value, list)
    strings = [item for item in value if isinstance(item, str)]
    assert len(strings) == len(value)
    return strings
