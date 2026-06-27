from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime.spine_contract import runtime_spine_contract
from sim_agent.schemas._parse import JsonMap


HARDENING_EVIDENCE_DIR: Final = "runtime-spine-hardening-20260627"
REQUIRED_TASK_IDS: Final = tuple(range(1, 18))
EVIDENCE_EVENTS: Final = {
    "red": "RedEvidence",
    "green": "GreenEvidence",
    "qa": "QAEvidence",
    "adversarial": "AdversarialVerify",
}
BLOCKER_RESOLVED_EVENTS: Final = frozenset({"BlockerResolved", "ResolvedBlocker"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence-root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    evidence_root = Path(args.evidence_root).resolve() if args.evidence_root else None
    payload = audit_runtime_spines(root, evidence_root=evidence_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"runtime_spine_audit_path={out_path}")
    print(f"runtime_spines={payload['summary']['total_spines']}")
    print(f"gap_open={payload['summary']['gap_open']}")
    print(f"readiness_status={payload['status']}")
    return 0


def audit_runtime_spines(root: Path, *, evidence_root: Path | None = None) -> JsonMap:
    contract = runtime_spine_contract()
    resolved_evidence_root = evidence_root or _default_evidence_root(root)
    evidence_audit = _audit_evidence(resolved_evidence_root)
    readiness_failures = evidence_audit.failure_codes
    spines = {
        spine.spine_id: {
            **spine.to_json(),
            "readiness_status": "incomplete" if readiness_failures else spine.status.value,
            "readiness_failures": readiness_failures,
            "detectors": _detectors(root, spine.spine_id),
        }
        for spine in contract.spines
    }
    required_failures = _required_detector_failures(spines)
    complete_spines = 0 if readiness_failures or required_failures else sum(
        1 for spine in contract.spines if spine.status.value == "complete"
    )
    return {
        "status": "incomplete" if readiness_failures or required_failures else "complete",
        "root": str(root),
        "evidence_root": str(resolved_evidence_root),
        "contract_version": contract.version,
        "summary": {
            "total_spines": len(contract.spines),
            "complete_spines": complete_spines,
            "gap_open": sum(1 for spine in contract.spines if spine.status.value == "gap_open"),
            "required_detector_failure_count": sum(len(failures) for failures in required_failures.values()),
            "required_detector_failures": required_failures,
            "readiness_failure_count": len(readiness_failures),
            "readiness_failure_codes": readiness_failures,
            "open_blockers": evidence_audit.open_blockers,
        },
        "evidence": evidence_audit.to_json(),
        "spines": spines,
    }


@dataclass(frozen=True, slots=True)
class EvidenceAudit:
    evidence_root: Path
    ledger_path: Path
    failure_codes: list[str]
    open_blockers: list[str]

    def to_json(self) -> JsonMap:
        return {
            "evidence_root": str(self.evidence_root),
            "ledger_path": str(self.ledger_path),
            "failure_codes": self.failure_codes,
            "open_blockers": self.open_blockers,
        }


def _required_detector_failures(spines: dict[str, JsonMap]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for spine_id, spine_payload in spines.items():
        if spine_payload.get("status") != "complete":
            continue
        detectors = spine_payload.get("detectors")
        if not isinstance(detectors, dict):
            failures[spine_id] = ["required_detectors_missing"]
            continue
        missing = [name for name, value in detectors.items() if name.startswith("required_") and value is not True]
        if missing:
            failures[spine_id] = missing
    return failures


def _audit_evidence(evidence_root: Path) -> EvidenceAudit:
    ledger_path = evidence_root / "ledger.jsonl"
    records = _read_ledger_records(ledger_path)
    failure_codes: list[str] = []
    if not records:
        failure_codes.append("ledger_missing_or_empty")
    if any(record.get("event") == "LedgerParseError" for record in records):
        failure_codes.append("ledger_malformed_jsonl")
    for task_id in REQUIRED_TASK_IDS:
        failure_codes.extend(_task_failure_codes(task_id, records, evidence_root))
    open_blockers = _open_blockers(records, evidence_root)
    failure_codes.extend(f"unresolved_blocker_{blocker}" for blocker in open_blockers)
    return EvidenceAudit(
        evidence_root=evidence_root,
        ledger_path=ledger_path,
        failure_codes=sorted(set(failure_codes)),
        open_blockers=open_blockers,
    )


def _read_ledger_records(ledger_path: Path) -> list[JsonMap]:
    if not ledger_path.is_file():
        return []
    records: list[JsonMap] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError:
            records.append({"event": "LedgerParseError"})
            continue
        if isinstance(raw_record, dict):
            records.append(raw_record)
    return records


def _task_failure_codes(task_id: int, records: list[JsonMap], evidence_root: Path) -> list[str]:
    failures: list[str] = []
    task_records = [record for record in records if record.get("task") == task_id]
    for label, event in EVIDENCE_EVENTS.items():
        event_records = [record for record in task_records if record.get("event") == event]
        if not event_records:
            failures.append(f"task_{task_id}_missing_{label}_evidence")
            continue
        if not any(_evidence_record_is_fresh(record, evidence_root) for record in event_records):
            failures.append(f"task_{task_id}_stale_{label}_evidence")
    if not any(record.get("event") == "DoneClaim" for record in task_records):
        failures.append(f"task_{task_id}_missing_done_claim")
    return failures


def _evidence_record_is_fresh(record: JsonMap, evidence_root: Path) -> bool:
    path_value = record.get("path")
    if not isinstance(path_value, str):
        return False
    evidence_path = _evidence_path(evidence_root, path_value)
    if evidence_path is None:
        return False
    if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
        return False
    sha256_value = record.get("sha256")
    if not isinstance(sha256_value, str):
        return False
    if len(sha256_value) != 64 or not all(character in "0123456789abcdefABCDEF" for character in sha256_value):
        return False
    return hashlib.sha256(evidence_path.read_bytes()).hexdigest() == sha256_value


def _evidence_path(evidence_root: Path, path_value: str) -> Path | None:
    path = Path(path_value)
    if path.is_absolute():
        return None
    marker = f".omo/evidence/{HARDENING_EVIDENCE_DIR}/"
    if path_value.startswith(marker):
        path = Path(path_value.removeprefix(marker))
    try:
        resolved_root = evidence_root.resolve()
        resolved_path = (resolved_root / path).resolve()
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved_path


def _open_blockers(records: list[JsonMap], evidence_root: Path) -> list[str]:
    blockers: set[str] = set()
    for record in records:
        event = record.get("event")
        tokens = _record_task_tokens(record)
        if event == "IntegrationBlocker":
            blockers.update(tokens)
        if event in BLOCKER_RESOLVED_EVENTS and _evidence_record_is_fresh(record, evidence_root):
            blockers.difference_update(tokens)
    return sorted(blockers)


def _record_task_tokens(record: JsonMap) -> set[str]:
    value = record.get("task", record.get("tasks"))
    if isinstance(value, list):
        return {str(item) for item in value}
    if value is None:
        return set()
    return {str(value)}


def _default_evidence_root(root: Path) -> Path:
    return root.resolve().parents[1] / ".omo" / "evidence" / HARDENING_EVIDENCE_DIR


def _detectors(root: Path, spine_id: str) -> JsonMap:
    files = _runtime_files(root)
    detectors: dict[str, bool] = {
        "contract_defined": True,
        "doc_contract_present": (root / "docs" / "runtime_spine_contract.md").is_file(),
    }
    if spine_id == "provider_transport":
        detectors["fixed_responses_endpoint"] = "/v1/responses" in files.provider_model
    if spine_id == "agent_session":
        detectors["agent_session_contract_defined"] = "class AsaAgentSession" in files.agent_loop_contract
        detectors["agent_session_exported"] = "AsaAgentSession" in files.agent_loop
        detectors["mutable_session_history"] = "messages: list[JsonMap]" in files.agent_loop_contract
    if spine_id == "agent_loop":
        detectors["required_model_turn_bridge"] = (
            "complete_turn = getattr(model, \"complete_turn\", None)" in files.agent_loop
            and "return ModelTurnResult(selected_tools=model.choose_tools(session, tool_schemas))" in files.agent_loop
        )
        detectors["required_tool_results_appended"] = "self.session.append_tool_result(result)" in files.agent_loop
        detectors["required_tool_result_continuation_gate"] = "supports_tool_result_continuation(self.model)" in files.agent_loop
        detectors["required_model_tool_events"] = (
            "RuntimeEventType.MODEL_START" in files.agent_loop
            and "RuntimeEventType.MODEL_DELTA" in files.agent_loop
            and "RuntimeEventType.TOOL_START" in files.agent_loop
            and "RuntimeEventType.TOOL_END" in files.agent_loop
        )
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
        self.agent_loop_contract = _read(root / "sim_agent" / "agents_sdk_runtime" / "agent_loop_contract.py")
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
