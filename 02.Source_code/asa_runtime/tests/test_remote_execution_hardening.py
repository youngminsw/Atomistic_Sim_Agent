from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from sim_agent.compute import (
    ComputePolicyError,
    run_remote_capability_probe,
    run_remote_chain,
    run_remote_execution_plan,
)
from sim_agent.compute import remote_capability_runner, remote_chain_runner, remote_plan_runner


def test_remote_plan_rejects_unapproved_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a model-controlled absolute plan outside ASA-approved remote roots.
    plan_path = tmp_path / "model_payload_plan.json"
    plan_path.write_text(
        json.dumps({"all_commands": ["echo should-not-run"]}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def spy_run_command(
        command: str,
        cwd: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            args=(command, cwd, timeout_s),
            returncode=0,
            stdout="ran",
            stderr="",
        )

    monkeypatch.setattr(remote_plan_runner, "_run_command", spy_run_command)

    # When / Then: provenance blocks before shell/subprocess execution.
    with pytest.raises(ComputePolicyError, match="remote_manifest_unapproved_root"):
        run_remote_execution_plan(plan_path, timeout_s=5)

    assert calls == []


def test_approved_remote_manifest_runs_in_approved_root(tmp_path: Path) -> None:
    # Given: an ASA-generated capability manifest and script under output_dir/remote.
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    remote_root = output_root / "remote"
    source_root.mkdir()
    remote_root.mkdir(parents=True)
    script_path = remote_root / "probe.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cat > worker_capability.json <<'JSON'\n"
        '{"ok": true, "gate_status": "worker_capability_ready"}\n'
        "JSON\n"
        "echo remote_capability_probe_done=true\n",
        encoding="utf-8",
    )
    manifest_path = remote_root / "capability.json"
    _write_manifest(
        manifest_path,
        source_root,
        output_root,
        {
            "kind": "remote_capability_probe",
            "probe_script": "probe.sh",
            "script_sha256": _file_hash(script_path),
            "expected_output": "worker_capability.json",
        },
    )

    # When: the approved manifest is executed.
    result = run_remote_capability_probe(manifest_path, timeout_s=5)

    # Then: execution succeeds and writes only within the approved output root.
    assert result.ok is True
    assert result.payload["probe_status"] == "remote_capability_ready"
    assert Path(str(result.payload["expected_output"])).is_relative_to(output_root)


def test_remote_script_and_output_paths_cannot_escape_approved_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: approved roots plus malicious relative paths and a symlink escape.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target_script = outside / "escape.sh"
    target_script.write_text("echo should-not-run\n", encoding="utf-8")
    symlink_script = remote_root / "escape.sh"
    symlink_script.symlink_to(target_script)
    script_calls: list[Path] = []

    def spy_run_script(
        script_path: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        script_calls.append(script_path)
        return subprocess.CompletedProcess(args=(script_path, timeout_s), returncode=0)

    monkeypatch.setattr(remote_capability_runner, "_run_script", spy_run_script)
    symlink_manifest = remote_root / "symlink_probe.json"
    _write_manifest(
        symlink_manifest,
        source_root,
        output_root,
        {
            "kind": "remote_capability_probe",
            "probe_script": "escape.sh",
            "script_sha256": _file_hash(target_script),
            "expected_output": "worker_capability.json",
        },
    )

    # When / Then: a symlink script escape blocks before subprocess execution.
    with pytest.raises(ComputePolicyError, match="remote_path_escape"):
        run_remote_capability_probe(symlink_manifest, timeout_s=5)
    assert script_calls == []

    safe_script = remote_root / "safe.sh"
    safe_script.write_text("echo should-not-run\n", encoding="utf-8")
    output_escape_manifest = remote_root / "output_escape_probe.json"
    _write_manifest(
        output_escape_manifest,
        source_root,
        output_root,
        {
            "kind": "remote_capability_probe",
            "probe_script": "safe.sh",
            "script_sha256": _file_hash(safe_script),
            "expected_output": "../../outside/worker_capability.json",
        },
    )

    # When / Then: output paths outside output_root also block before subprocess.
    with pytest.raises(ComputePolicyError, match="remote_output_root_violation"):
        run_remote_capability_probe(output_escape_manifest, timeout_s=5)
    assert script_calls == []


def test_remote_manifest_requires_provenance_and_matching_hash(tmp_path: Path) -> None:
    # Given: a manifest under an approved root but without ASA provenance.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    commands = ("echo should-not-run",)
    missing_provenance = remote_root / "missing_provenance.json"
    missing_provenance.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "all_commands": list(commands),
                "plan_sha256": _commands_hash(commands),
            }
        ),
        encoding="utf-8",
    )

    # When / Then: missing created_by/kind/schema_version is a typed blocker.
    with pytest.raises(ComputePolicyError, match="remote_manifest_missing_provenance"):
        run_remote_execution_plan(missing_provenance, timeout_s=5)

    mismatched_hash = remote_root / "mismatched_hash.json"
    _write_manifest(
        mismatched_hash,
        source_root,
        output_root,
        {
            "kind": "remote_execution_plan",
            "all_commands": list(commands),
            "plan_sha256": "0" * 64,
        },
    )

    # When / Then: command payload hash mismatches are rejected before execution.
    with pytest.raises(ComputePolicyError, match="remote_manifest_hash_mismatch"):
        run_remote_execution_plan(mismatched_hash, timeout_s=5)


def test_remote_stdout_and_stderr_secret_tails_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an approved chain manifest whose process emits secret-looking tails.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    script_path = remote_root / "chain.sh"
    script_path.write_text("echo placeholder\n", encoding="utf-8")
    manifest_path = remote_root / "chain.json"
    _write_manifest(
        manifest_path,
        source_root,
        output_root,
        {
            "kind": "remote_execution_chain",
            "executable_script": "chain.sh",
            "script_sha256": _file_hash(script_path),
            "stage_ids": ["01-md"],
        },
    )

    def secret_run_script(
        script_path: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=(script_path, timeout_s),
            returncode=0,
            stdout=(
                "stage_done=01-md\n"
                "api_key=sk-secret-value\n"
                "remote_execution_chain_done=true\n"
            ),
            stderr="token=super-secret\n",
        )

    monkeypatch.setattr(remote_chain_runner, "_run_script", secret_run_script)

    # When: a subprocess result contains secret-looking tails.
    result = run_remote_chain(manifest_path, timeout_s=5)

    # Then: raw secrets are absent and the redaction blocker is explicit.
    assert result.ok is False
    assert "remote_secret_tail_redacted" in result.payload["blockers"]
    assert "sk-secret-value" not in result.payload["stdout_tail"]
    assert "super-secret" not in result.payload["stderr_tail"]


def test_remote_script_tamper_after_hash_blocks_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an approved probe manifest whose script changes after hashing.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    script_path = remote_root / "probe.sh"
    script_path.write_text("echo original\n", encoding="utf-8")
    manifest_path = remote_root / "probe.json"
    _write_manifest(
        manifest_path,
        source_root,
        output_root,
        {
            "kind": "remote_capability_probe",
            "probe_script": "probe.sh",
            "script_sha256": _file_hash(script_path),
            "expected_output": "worker_capability.json",
        },
    )
    script_path.write_text("echo tampered\n", encoding="utf-8")
    script_calls: list[Path] = []

    def spy_run_script(
        script_path: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        script_calls.append(script_path)
        return subprocess.CompletedProcess(args=(script_path, timeout_s), returncode=0)

    monkeypatch.setattr(remote_capability_runner, "_run_script", spy_run_script)

    # When / Then: hash mismatch blocks before the runner executes the script.
    with pytest.raises(ComputePolicyError, match="remote_manifest_hash_mismatch"):
        run_remote_capability_probe(manifest_path, timeout_s=5)
    assert script_calls == []


def test_remote_script_crlf_hash_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a CRLF script whose manifest hash matches the unnormalized bytes.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    script_path = remote_root / "chain.sh"
    original_bytes = b"#!/usr/bin/env bash\r\necho stage_done=01-md\r\n"
    script_path.write_bytes(original_bytes)
    manifest_path = remote_root / "chain.json"
    _write_manifest(
        manifest_path,
        source_root,
        output_root,
        {
            "kind": "remote_execution_chain",
            "executable_script": "chain.sh",
            "script_sha256": _file_hash(script_path),
            "stage_ids": ["01-md"],
        },
    )
    script_calls: list[Path] = []

    def spy_run_script(
        script_path: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        script_calls.append(script_path)
        return subprocess.CompletedProcess(args=(script_path, timeout_s), returncode=0)

    monkeypatch.setattr(remote_chain_runner, "_run_script", spy_run_script)

    # When / Then: newline normalization is rejected before execution and bytes remain intact.
    with pytest.raises(ComputePolicyError, match="remote_script_newline_not_normalized"):
        run_remote_chain(manifest_path, timeout_s=5)
    assert script_calls == []
    assert script_path.read_bytes() == original_bytes


def test_remote_chain_misleading_success_output_still_requires_stage_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a chain script whose process claims global success but omits a stage marker.
    source_root, output_root, remote_root = _approved_roots(tmp_path)
    script_path = remote_root / "chain.sh"
    script_path.write_text("echo placeholder\n", encoding="utf-8")
    manifest_path = remote_root / "chain.json"
    _write_manifest(
        manifest_path,
        source_root,
        output_root,
        {
            "kind": "remote_execution_chain",
            "executable_script": "chain.sh",
            "script_sha256": _file_hash(script_path),
            "stage_ids": ["01-md", "02-lammps"],
        },
    )

    def misleading_run_script(
        script_path: Path,
        timeout_s: float | None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=(script_path, timeout_s),
            returncode=0,
            stdout="stage_done=01-md\nremote_execution_chain_done=true\n",
            stderr="",
        )

    monkeypatch.setattr(remote_chain_runner, "_run_script", misleading_run_script)

    # When: stdout contains the completion marker but not every stage marker.
    result = run_remote_chain(manifest_path, timeout_s=5)

    # Then: the runner reports incomplete execution instead of trusting the success-looking tail.
    assert result.ok is False
    assert "remote_chain_stage_incomplete" in result.payload["blockers"]
    assert result.payload["missing_stage_ids"] == ["02-lammps"]


def _approved_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    remote_root = output_root / "remote"
    source_root.mkdir()
    remote_root.mkdir(parents=True)
    return (source_root, output_root, remote_root)


def _write_manifest(
    manifest_path: Path,
    source_root: Path,
    output_root: Path,
    payload: dict[str, str | list[str]],
) -> None:
    manifest = {
        "schema_version": 1,
        "created_by": "asa_runtime",
        "source_root": str(source_root),
        "output_root": str(output_root),
        **payload,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _commands_hash(commands: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(commands),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
