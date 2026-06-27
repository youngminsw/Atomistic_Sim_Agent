from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_runtime_spines import audit_runtime_spines, _evidence_record_is_fresh
from scripts.audit_scope_fidelity import audit_scope_paths


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_audit_rejects_static_complete_when_blocker_evidence_missing(tmp_path: Path) -> None:
    # Given: the contract reports static complete status for every runtime spine.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    (evidence_root / "ledger.jsonl").write_text(
        json.dumps({"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"}) + "\n",
        encoding="utf-8",
    )

    # When: the audit evaluates the current worktree evidence ledger.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: completion is withheld until evidence and blocker resolution exist.
    assert audit["status"] == "incomplete"
    assert audit["summary"]["complete_spines"] == 0
    assert audit["summary"]["readiness_failure_count"] > 0
    failure_codes = set(audit["summary"]["readiness_failure_codes"])
    assert any(
        code.endswith("_missing_red_evidence")
        or code.endswith("_stale_red_evidence")
        or code.startswith("unresolved_blocker_")
        for code in failure_codes
    )
    assert "unresolved_blocker_full-pytest" in failure_codes


def test_audit_accepts_complete_fresh_evidence_set(tmp_path: Path) -> None:
    # Given: every planned task has red, green, QA, done, adversarial, and blocker-resolution evidence.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    ledger_path = evidence_root / "ledger.jsonl"
    records: list[dict[str, str | int | list[int]]] = []
    for task_id in range(1, 18):
        red = _write_evidence(evidence_root, task_id, "red")
        green = _write_evidence(evidence_root, task_id, "green")
        qa = _write_evidence(evidence_root, task_id, "qa")
        adversarial = _write_evidence(evidence_root, task_id, "adversarial")
        records.extend(
            [
                _evidence_record(task_id, "RedEvidence", red),
                _evidence_record(task_id, "GreenEvidence", green),
                _evidence_record(task_id, "QAEvidence", qa),
                {"task": task_id, "event": "DoneClaim", "summary": f"task {task_id} done"},
                _evidence_record(task_id, "AdversarialVerify", adversarial),
            ]
        )
    records.append({"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"})
    resolved = _write_evidence(evidence_root, 17, "full-pytest-resolved")
    records.append(_evidence_record("full-pytest", "BlockerResolved", resolved))
    ledger_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")

    # When: the audit evaluates that evidence set.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: every spine may be reported complete because evidence is fresh and blockers are closed.
    assert audit["status"] == "complete"
    assert audit["summary"]["complete_spines"] == audit["summary"]["total_spines"]
    assert audit["summary"]["readiness_failure_count"] == 0
    assert audit["summary"]["open_blockers"] == []


def test_audit_reports_typed_reasons_for_missing_and_stale_evidence(tmp_path: Path) -> None:
    # Given: one task points at stale evidence and omits the rest.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    stale = _write_evidence(evidence_root, 1, "red")
    ledger_path = evidence_root / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "task": 1,
                "event": "RedEvidence",
                "path": f".omo/evidence/runtime-spine-hardening-20260627/{stale.name}",
                "sha256": "not-the-current-hash",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the audit evaluates incomplete evidence.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: it reports stable machine-readable reasons instead of trusting static status.
    failure_codes = set(audit["summary"]["readiness_failure_codes"])
    assert "task_1_stale_red_evidence" in failure_codes
    assert "task_1_missing_green_evidence" in failure_codes
    assert "task_1_missing_qa_evidence" in failure_codes


def test_audit_rejects_hashless_task_evidence(tmp_path: Path) -> None:
    # Given: task evidence points to a non-empty in-root artifact but omits sha256.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    red = _write_evidence(evidence_root, 1, "red")
    (evidence_root / "ledger.jsonl").write_text(json.dumps({"task": 1, "event": "RedEvidence", "path": _ledger_path(red)}, sort_keys=True) + "\n", encoding="utf-8")

    # When: the runtime spine audit evaluates the hashless evidence record.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: hashless evidence is stale even though the artifact exists in-root.
    assert _evidence_record_is_fresh({"event": "RedEvidence", "path": red.name}, evidence_root) is False
    assert _evidence_record_is_fresh({"event": "RedEvidence", "path": red.name, "sha256": 123}, evidence_root) is False
    assert "task_1_stale_red_evidence" in audit["summary"]["readiness_failure_codes"]


def test_audit_reports_malformed_ledger_evidence(tmp_path: Path) -> None:
    # Given: a ledger contains malformed JSONL next to a plausible success line.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    ledger_path = evidence_root / "ledger.jsonl"
    ledger_path.write_text(
        '{"task": 1, "event": "DoneClaim", "summary": "looks green"}\n'
        '{"task": 1, "event": "RedEvidence"\n',
        encoding="utf-8",
    )

    # When: the audit evaluates that evidence.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: malformed evidence is reported instead of trusting misleading success text.
    failure_codes = set(audit["summary"]["readiness_failure_codes"])
    assert "ledger_malformed_jsonl" in failure_codes
    assert audit["status"] == "incomplete"


def test_audit_accepts_resolved_blocker_event_alias(tmp_path: Path) -> None:
    # Given: current ledger evidence uses the ResolvedBlocker event spelling.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    ledger_path = evidence_root / "ledger.jsonl"
    records: list[dict[str, str | int]] = []
    for task_id in range(1, 18):
        red = _write_evidence(evidence_root, task_id, "red")
        green = _write_evidence(evidence_root, task_id, "green")
        qa = _write_evidence(evidence_root, task_id, "qa")
        adversarial = _write_evidence(evidence_root, task_id, "adversarial")
        records.extend(
            [
                _evidence_record(task_id, "RedEvidence", red),
                _evidence_record(task_id, "GreenEvidence", green),
                _evidence_record(task_id, "QAEvidence", qa),
                _evidence_record(task_id, "AdversarialVerify", adversarial),
                {"task": task_id, "event": "DoneClaim", "summary": f"task {task_id} done"},
            ]
        )
    records.append({"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"})
    resolved = _write_evidence(evidence_root, "full-pytest", "resolved")
    records.append(_evidence_record("full-pytest", "ResolvedBlocker", resolved))
    ledger_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")

    # When: the runtime spine audit evaluates blocker resolution.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: the alias closes the blocker instead of leaving stale open evidence.
    assert audit["summary"]["open_blockers"] == []
    assert "unresolved_blocker_full-pytest" not in audit["summary"]["readiness_failure_codes"]


def test_audit_requires_fresh_blocker_resolution_evidence(tmp_path: Path) -> None:
    # Given: a blocker has a resolution claim without a fresh artifact.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    ledger_path = evidence_root / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in [
                {"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"},
                {"task": "full-pytest", "event": "ResolvedBlocker", "summary": "claimed only"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the runtime spine audit evaluates blocker state.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: the blocker stays open instead of trusting a pathless resolution claim.
    assert audit["summary"]["open_blockers"] == ["full-pytest"]
    assert "unresolved_blocker_full-pytest" in audit["summary"]["readiness_failure_codes"]


def test_audit_keeps_blocker_open_for_hashless_resolution_evidence(tmp_path: Path) -> None:
    # Given: a blocker resolution points to an in-root artifact but omits sha256.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    resolved = _write_evidence(evidence_root, "full-pytest", "resolved")
    records = [
        {"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"},
        {"task": "full-pytest", "event": "ResolvedBlocker", "path": _ledger_path(resolved)},
    ]
    (evidence_root / "ledger.jsonl").write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")

    # When: the runtime spine audit evaluates blocker state.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: the hashless resolution artifact is stale and cannot close the blocker.
    assert audit["summary"]["open_blockers"] == ["full-pytest"]
    assert "unresolved_blocker_full-pytest" in audit["summary"]["readiness_failure_codes"]


def test_audit_rejects_absolute_evidence_record_paths(tmp_path: Path) -> None:
    # Given: an evidence record points at a real absolute path outside the evidence root.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    escape = tmp_path / "absolute-escape.txt"
    escape.write_text("outside evidence root\n", encoding="utf-8")
    absolute_record = _valid_sha_outside_root_record("RedEvidence", str(escape), escape)

    # When/Then: freshness rejects the escape despite a valid sha256 for the host file.
    assert absolute_record["sha256"] == hashlib.sha256(escape.read_bytes()).hexdigest()
    assert _evidence_record_is_fresh(absolute_record, evidence_root) is False


def test_audit_rejects_parent_traversal_evidence_record_paths(tmp_path: Path) -> None:
    # Given: traversal would otherwise resolve to a real file outside the evidence root.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    escape = tmp_path / "escape.txt"
    escape.write_text("outside evidence root\n", encoding="utf-8")
    traversal_record = _valid_sha_outside_root_record("RedEvidence", "../escape.txt", escape)

    # When/Then: freshness rejects the path escape despite a valid sha256 outside the root.
    assert traversal_record["sha256"] == hashlib.sha256(escape.read_bytes()).hexdigest()
    assert _evidence_record_is_fresh(traversal_record, evidence_root) is False


def test_audit_rejects_symlink_evidence_record_escapes(tmp_path: Path) -> None:
    # Given: an in-root filename is a symlink to content outside the evidence root.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    escape = tmp_path / "escape.txt"
    escape.write_text("outside evidence root\n", encoding="utf-8")
    link = evidence_root / "linked-escape.txt"
    link.symlink_to(escape)
    symlink_record = _valid_sha_outside_root_record("RedEvidence", link.name, escape)

    # When/Then: freshness rejects resolved paths that leave the root despite a valid target sha256.
    assert symlink_record["sha256"] == hashlib.sha256(escape.read_bytes()).hexdigest()
    assert _evidence_record_is_fresh(symlink_record, evidence_root) is False


def test_audit_keeps_blocker_open_when_resolution_path_escapes_evidence_root(tmp_path: Path) -> None:
    # Given: a blocker resolution points outside the configured evidence root.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    escape = tmp_path / "resolved.txt"
    escape.write_text("spoofed blocker resolution\n", encoding="utf-8")
    ledger_path = evidence_root / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in [
                {"task": "full-pytest", "event": "IntegrationBlocker", "summary": "blocked"},
                {
                    "task": "full-pytest",
                    **_valid_sha_outside_root_record("ResolvedBlocker", "../resolved.txt", escape),
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the runtime spine audit evaluates blocker state.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: escaped evidence with a valid outside-root sha256 does not clear the blocker.
    assert audit["summary"]["open_blockers"] == ["full-pytest"]
    assert "unresolved_blocker_full-pytest" in audit["summary"]["readiness_failure_codes"]


def test_audit_reopens_later_blocker_after_resolution(tmp_path: Path) -> None:
    # Given: a blocker is resolved and then a later run reports the same blocker again.
    evidence_root = tmp_path / "runtime-spine-hardening-20260627"
    evidence_root.mkdir()
    resolved = _write_evidence(evidence_root, "full-pytest", "resolved")
    ledger_path = evidence_root / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in [
                {"task": "full-pytest", "event": "IntegrationBlocker", "summary": "first failure"},
                _evidence_record("full-pytest", "ResolvedBlocker", resolved),
                {"task": "full-pytest", "event": "IntegrationBlocker", "summary": "regressed later"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # When: the runtime spine audit evaluates blocker state chronologically.
    audit = audit_runtime_spines(SOURCE_ROOT, evidence_root=evidence_root)

    # Then: the later blocker is open and must be resolved by newer evidence.
    assert audit["summary"]["open_blockers"] == ["full-pytest"]
    assert "unresolved_blocker_full-pytest" in audit["summary"]["readiness_failure_codes"]


def test_scope_fidelity_accepts_approved_runtime_hardening_paths() -> None:
    # Given: the current hardening plan allows runtime code, tests, scripts, and gateway edits.
    changed_paths = [
        "02.Source_code/asa_runtime/sim_agent/agents_sdk_runtime/provider_transport.py",
        "02.Source_code/asa_runtime/sim_agent/knowledge/mcp_manager.py",
        "02.Source_code/asa_runtime/sim_agent/compute/remote_plan.py",
        "02.Source_code/asa_runtime/model_gateway/src/gateway/server.ts",
        "02.Source_code/asa_runtime/model_gateway/test/gateway-server.test.ts",
        "02.Source_code/asa_runtime/tests/test_provider_transport_hardening.py",
        "02.Source_code/asa_runtime/scripts/audit_scope_fidelity.py",
        ".omo/evidence/runtime-spine-hardening-20260627/task-8-mcp-env-red.txt",
    ]

    # When: scope fidelity audits those paths.
    payload = audit_scope_paths(SOURCE_ROOT, changed_paths)

    # Then: approved plan and ignored evidence paths do not require scope review.
    assert payload["status"] == "clean"
    assert payload["out_of_scope_paths"] == []
    assert payload["outside_runtime_violations"] == []


def test_scope_fidelity_rejects_root_bundle_and_domain_prompt_changes() -> None:
    # Given: a root bundle and domain-prompt edit are outside the approved hardening contract.
    changed_paths = [
        "worker_bundle.json",
        "02.Source_code/asa_runtime/sim_agent/agents/materials/system.md",
    ]

    # When: scope fidelity audits those paths.
    payload = audit_scope_paths(SOURCE_ROOT, changed_paths)

    # Then: both violations are reported explicitly.
    assert payload["status"] == "scope_review_required"
    assert payload["outside_runtime_violations"] == ["worker_bundle.json"]
    assert payload["domain_prompt_paths"] == ["sim_agent/agents/materials/system.md"]


def _write_evidence(evidence_root: Path, task_id: int | str, label: str) -> Path:
    path = evidence_root / f"task-{task_id}-{label}.txt"
    path.write_text(f"{task_id}:{label}:fresh\n", encoding="utf-8")
    return path


def _evidence_record(task_id: int | str, event: str, path: Path) -> dict[str, str | int]:
    return {
        "task": task_id,
        "event": event,
        "path": _ledger_path(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _valid_sha_outside_root_record(event: str, path_value: str, path: Path) -> dict[str, str]:
    return {
        "event": event,
        "path": path_value,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ledger_path(path: Path) -> str:
    return f".omo/evidence/runtime-spine-hardening-20260627/{path.name}"
