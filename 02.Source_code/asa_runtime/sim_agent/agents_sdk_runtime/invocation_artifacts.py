from __future__ import annotations

import json
from pathlib import Path

from sim_agent.schemas._parse import JsonMap

from .types import SkillInvocationResult


def write_skill_invocation_artifact(output_dir: Path, invocation: SkillInvocationResult) -> Path:
    path = output_dir / invocation.artifact_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skill_invocation_payload(invocation), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def skill_invocation_payload(invocation: SkillInvocationResult) -> JsonMap:
    return {
        "artifact_version": "asa_skill_invocation_v1",
        "agent_id": invocation.agent_id,
        "skill_id": invocation.skill_id,
        "status": invocation.status,
        "execution_status": invocation.execution_status,
        "domain_adapter": invocation.domain_adapter,
        "artifact_ref": invocation.artifact_ref,
        "contract": invocation.contract,
        "result": invocation.result,
    }
