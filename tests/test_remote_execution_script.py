from __future__ import annotations

import json
import sys
from pathlib import Path

from pytest import MonkeyPatch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import write_remote_execution_script_bundle
from sim_agent.compute.types import (
    RemoteExecutionChain,
    RemoteExecutionPlan,
    RemoteExecutionStage,
)
from sim_agent.schemas._parse import as_mapping


def test_remote_execution_script_manifest_survives_chmod_denied(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    def deny_chmod(_path: Path, _mode: int) -> None:
        raise OSError("chmod denied")

    monkeypatch.setattr(Path, "chmod", deny_chmod)
    bundle = write_remote_execution_script_bundle(
        _chain(),
        script_path=tmp_path / "remote_chain.sh",
        manifest_path=tmp_path / "remote_chain_manifest.json",
    )

    manifest = as_mapping(
        json.loads(bundle.manifest_path.read_text(encoding="utf-8")),
        "remote_chain_manifest",
    )
    assert bundle.script_path.exists()
    assert manifest["chmod_applied"] is False
    assert manifest["run_command"] == f"bash {bundle.script_path}"


def _chain() -> RemoteExecutionChain:
    plan = RemoteExecutionPlan(
        ssh_target="swym@10.24.12.85",
        ssh_port=55555,
        local_setup_commands=("mkdir -p artifacts",),
        remote_setup_commands=("ssh -p 55555 swym@10.24.12.85 mkdir",),
        upload_commands=("rsync source_payload.tar.gz remote:",),
        preflight_commands=("ssh remote tar -xzf source_payload.tar.gz",),
        execution_command=("ssh remote python3 02.Source_code/mss_agent/scripts/run.py"),
        download_commands=("rsync remote:artifact .",),
    )
    return RemoteExecutionChain(
        ssh_target="swym@10.24.12.85",
        ssh_port=55555,
        stages=(RemoteExecutionStage("01-smoke", "smoke", plan),),
    )
