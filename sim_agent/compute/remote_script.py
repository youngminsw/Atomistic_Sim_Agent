from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .types import RemoteExecutionChain


@dataclass(frozen=True, slots=True)
class RemoteExecutionScriptBundle:
    script_path: Path
    manifest_path: Path
    manifest_payload: JsonMap


def write_remote_execution_script_bundle(
    chain: RemoteExecutionChain,
    script_path: Path,
    manifest_path: Path,
) -> RemoteExecutionScriptBundle:
    script_text = _script_text(chain)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    chmod_applied = _try_set_executable(script_path)
    manifest_payload = _manifest_payload(chain, script_path, chmod_applied)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RemoteExecutionScriptBundle(
        script_path=script_path,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )


def _try_set_executable(script_path: Path) -> bool:
    try:
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    except OSError:
        return False
    return True


def _script_text(chain: RemoteExecutionChain) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "",
    ]
    for stage in chain.stages:
        lines.append(f"echo 'stage_start={stage.stage_id}'")
        lines.extend(stage.plan.all_commands)
        lines.append(f"echo 'stage_done={stage.stage_id}'")
        lines.append("")
    lines.append("echo 'remote_execution_chain_done=true'")
    return "\n".join(lines) + "\n"


def _manifest_payload(
    chain: RemoteExecutionChain,
    script_path: Path,
    chmod_applied: bool,
) -> JsonMap:
    return {
        "executable_script": str(script_path),
        "run_command": f"bash {script_path}",
        "chmod_applied": chmod_applied,
        "ssh_target": chain.ssh_target,
        "ssh_port": chain.ssh_port,
        "stage_count": len(chain.stages),
        "stage_ids": [stage.stage_id for stage in chain.stages],
        "command_count": len(chain.all_commands),
        "dry_run": False,
    }
