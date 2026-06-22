from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime.spine_contract import runtime_spine_contract
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    payload = audit_runtime_spines(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"runtime_spine_audit_path={out_path}")
    print(f"runtime_spines={payload['summary']['total_spines']}")
    print(f"gap_open={payload['summary']['gap_open']}")
    return 0


def audit_runtime_spines(root: Path) -> JsonMap:
    contract = runtime_spine_contract()
    spines = {
        spine.spine_id: {
            **spine.to_json(),
            "detectors": _detectors(root, spine.spine_id),
        }
        for spine in contract.spines
    }
    return {
        "status": "gap_contract_recorded",
        "root": str(root),
        "contract_version": contract.version,
        "summary": {
            "total_spines": len(contract.spines),
            "gap_open": sum(1 for spine in contract.spines if spine.status.value == "gap_open"),
        },
        "spines": spines,
    }


def _detectors(root: Path, spine_id: str) -> JsonMap:
    files = _runtime_files(root)
    detectors: dict[str, bool] = {
        "contract_defined": True,
        "doc_contract_present": (root / "docs" / "runtime_spine_contract.md").is_file(),
    }
    if spine_id == "provider_transport":
        detectors["fixed_responses_endpoint"] = "/v1/responses" in files.provider_model
    if spine_id == "agent_session":
        detectors["frozen_session_dto"] = "class AsaAgentSession" in files.agent_loop and "frozen=True" in files.agent_loop
    if spine_id == "agent_loop":
        detectors["one_shot_choose_tools"] = "choose_tools(self.session, tool_schemas)" in files.agent_loop
    if spine_id == "subagent_runtime":
        detectors["subagent_tool_defined"] = "subagent_task" in files.tools
    if spine_id == "context_resume":
        detectors["live_turn_uses_user_goal"] = "_agent_loop_session(handle, user_goal)" in files.live_turn
    if spine_id == "tool_runtime":
        detectors["placeholder_capabilities_present"] = "validate_simulation_request" in files.tools
    if spine_id == "tui_observability":
        detectors["semantic_direct_route_present"] = "agent_direct_route" in files.tui_chat
    if spine_id == "assembly":
        detectors["provider_payload_uses_current_goal"] = "session.user_goal" in files.provider_model
    return detectors


class _RuntimeFiles:
    def __init__(self, root: Path) -> None:
        self.agent_loop = _read(root / "sim_agent" / "agents_sdk_runtime" / "agent_loop.py")
        self.provider_model = _read(root / "sim_agent" / "agents_sdk_runtime" / "provider_tool_choice_model.py")
        self.live_turn = _read(root / "sim_agent" / "agent_runtime" / "live_agent_turn.py")
        self.tools = _read(root / "sim_agent" / "agent_harness" / "tools.py")
        self.tui_chat = _read(root / "sim_agent" / "cli" / "tui_chat.py")


def _runtime_files(root: Path) -> _RuntimeFiles:
    return _RuntimeFiles(root)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


if __name__ == "__main__":
    raise SystemExit(main())
